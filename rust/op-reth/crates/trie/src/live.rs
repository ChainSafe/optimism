//! Live trie collector for external proofs storage.

use crate::{
    provider::OpProofsStateProviderRef, state::LiveTrieState, BlockStateDiff,
    OpProofsStorageError, OpProofsStore, OpProofsProviderRO, persistence::{LiveTriePersistenceHandle, PersistenceStatus},
    OpProofStoragePruner,
};
#[cfg(feature = "metrics")]
use crate::metrics::LiveMetrics;
use alloy_eips::{eip1898::BlockWithParent, NumHash};
use crossbeam_channel::{bounded, RecvTimeoutError};
use reth_evm::{execute::Executor, ConfigureEvm};
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, RecoveredBlock};
use reth_provider::{
    BlockHashReader, DatabaseProviderFactory, HashedPostStateProvider, StateProviderFactory, StateReader, StateRootProvider
};
use reth_revm::database::StateProviderDatabase;
use reth_trie_common::{updates::TrieUpdatesSorted, HashedPostStateSorted};
use std::{sync::Arc, time::{Duration, Instant}};
use tracing::{error, info};

/// Default number of blocks to keep in memory before persisting.
pub const DEFAULT_PERSISTENCE_THRESHOLD: u64 = 5;

/// Default number of blocks where we block execution to allow persistence to catch up.
pub const DEFAULT_BACKPRESSURE_THRESHOLD: u64 = 10;

/// Default timeout for waiting on a persistence save operation (in seconds).
pub const DEFAULT_PERSISTENCE_TIMEOUT_SECS: u64 = 60;

/// Live trie collector for external proofs storage.
#[derive(Debug)]
pub struct LiveTrieCollector<Evm, Provider, Store>
where
    Evm: ConfigureEvm,
    Provider: StateReader + DatabaseProviderFactory + StateProviderFactory,
{
    evm_config: Evm,
    provider: Provider,
    storage: Store,
    memory: LiveTrieState,

    /// Number of blocks to keep in memory before persisting.
    persistence_threshold: u64,
    /// Number of blocks to keep in memory limit (backpressure).
    backpressure_threshold: u64,
    persistence_handle: LiveTriePersistenceHandle,
    /// Tracks if a background persistence task is currently running.
    persistence_status: Arc<PersistenceStatus>,

    #[cfg(feature = "metrics")]
    metrics: LiveMetrics,
}

impl<Evm, Provider, Store> LiveTrieCollector<Evm, Provider, Store>
where
    Evm: ConfigureEvm,
    Provider: BlockHashReader + StateReader + DatabaseProviderFactory + StateProviderFactory + Clone + 'static,
    Store: OpProofsStore + Clone + 'static,
{
    /// Create a new live trie collector.
    pub fn new(
        evm_config: Evm,
        provider: Provider,
        storage: Store,
        pruner: OpProofStoragePruner<Store, Provider>,
    ) -> Self {
        let persistence_handle = LiveTriePersistenceHandle::spawn(pruner, storage.clone());
        Self {
            evm_config,
            provider,
            storage,
            memory: LiveTrieState::new(),

            persistence_threshold: DEFAULT_PERSISTENCE_THRESHOLD,
            backpressure_threshold: DEFAULT_BACKPRESSURE_THRESHOLD,
            persistence_handle,
            persistence_status: Arc::new(PersistenceStatus::new()),

            #[cfg(feature = "metrics")]
            metrics: LiveMetrics::new_with_labels(&[] as &[(&str, &str)]),
        }
    }

    /// Set the persistence threshold (number of blocks to keep in memory before persisting).
    pub fn with_persistence_threshold(mut self, threshold: u64) -> Self {
        self.persistence_threshold = threshold;
        self
    }

    /// Set the backpressure threshold (number of blocks before execution blocks).
    pub fn with_backpressure_threshold(mut self, threshold: u64) -> Self {
        self.backpressure_threshold = threshold;
        self
    }

    /// Execute a block and store the updates in the in-memory buffer.
    pub fn execute_and_store_block_updates(
        &self,
        block: &RecoveredBlock<BlockTy<Evm::Primitives>>,
    ) -> Result<(), OpProofsStorageError> {
        let start = Instant::now();

        // Check if we have the parent state
        let tip = self.get_tip()?;
        let parent_block_number = block.number() - 1;

        if block.parent_hash() != tip.hash {
            return Err(OpProofsStorageError::OutOfOrder {
                block_number: block.number(),
                parent_block_hash: block.parent_hash(),
                latest_block_hash: tip.hash,
            });
        }

        let block_ref =
            BlockWithParent::new(block.parent_hash(), NumHash::new(block.number(), block.hash()));

        let inner_provider = OpProofsStateProviderRef::new(
            self.provider.state_by_block_hash(block.parent_hash())?,
            self.storage.provider_ro()?,
            parent_block_number,
        );

        // 2. Wrap it with memory overlay using LiveTrieState
        // This gathers all buffered blocks required to build the state on top of disk
        let state_provider = self.memory.state_provider(block.parent_hash(), inner_provider);

        // 3. Execute block
        let db = StateProviderDatabase::new(&state_provider);
        let block_executor = self.evm_config.batch_executor(db);

        let execution_result = block_executor.execute(&(*block).clone())?;

        let execution_duration = start.elapsed();

        // 4. Calculate state root
        let hashed_state = state_provider.hashed_post_state(&execution_result.state);
        let (state_root, trie_updates) =
            state_provider.state_root_with_updates(hashed_state.clone())?;

        let state_root_duration = start.elapsed() - execution_duration;

        // 5. Verify root
        if state_root != block.state_root() {
            return Err(OpProofsStorageError::StateRootMismatch {
                block_number: block.number(),
                current_state_hash: state_root,
                expected_state_hash: block.state_root(),
            });
        }

        // 6. Store Diff to Memory
        self.memory.insert(
            block_ref,
            BlockStateDiff {
                sorted_trie_updates: trie_updates.into_sorted(),
                sorted_post_state: hashed_state.into_sorted(),
            },
        );

        let total_duration = start.elapsed();

        #[cfg(feature = "metrics")]
        {
            self.metrics.total_duration_seconds.record(total_duration);
            self.metrics.execution_duration_seconds.record(execution_duration);
            self.metrics.state_root_duration_seconds.record(state_root_duration);
        }

        info!(
            block_number = block.number(),
            ?total_duration,
            ?execution_duration,
            ?state_root_duration,
            "Block executed and trie updates buffered successfully",
        );

        // Trigger persistence
        self.advance_persistence()?;

        Ok(())
    }

    /// Store trie updates for a given block.
    pub fn store_block_updates(
        &self,
        block: BlockWithParent,
        sorted_trie_updates: TrieUpdatesSorted,
        sorted_post_state: HashedPostStateSorted,
    ) -> Result<(), OpProofsStorageError> {
        let start = Instant::now();

        // Check if we have the parent state
        let tip = self.get_tip()?;

        if block.parent != tip.hash {
            return Err(OpProofsStorageError::OutOfOrder {
                block_number: block.block.number,
                parent_block_hash: block.parent,
                latest_block_hash: tip.hash,
            });
        }

        self.memory.insert(
            block,
            BlockStateDiff { sorted_trie_updates, sorted_post_state },
        );

        let total_duration = start.elapsed();

        #[cfg(feature = "metrics")]
        self.metrics.total_duration_seconds.record(total_duration);

        info!(
            block_number = block.block.number,
            ?total_duration,
            "Trie updates buffered successfully",
        );

        // Trigger persistence check
        self.advance_persistence()?;

        Ok(())
    }

    /// Handles chain reorganizations by replacing block updates after a common ancestor.
    ///
    /// This method removes all block updates after the latest common ancestor (the block before
    /// the first block in `new_blocks`) and replaces them with the updates from the provided new
    /// chain.
    ///
    /// # Arguments
    ///
    /// * `new_blocks` - A vector of references to `RecoveredBlock` instances representing the new
    ///   blocks to be added to the trie storage.
    pub fn unwind_and_store_block_updates(
        &self,
        block_updates: Vec<(BlockWithParent, Arc<TrieUpdatesSorted>, Arc<HashedPostStateSorted>)>,
    ) -> Result<(), OpProofsStorageError> {
        if block_updates.is_empty() {
            return Ok(());
        }

        let start = Instant::now();
        let first = &block_updates[0].0;
        // The common ancestor is one block before the first diverging block.
        let common_ancestor_number = first.block.number.saturating_sub(1);

        info!(
            target: "live-trie",
            reorg_depth = block_updates.len(),
            common_ancestor = common_ancestor_number,
            "Handling reorg: unwinding and buffering new path"
        );

        let unwind_start = Instant::now();
        // 1. Unwind Persistence (Disk)
        // `unwind_history` on the store removes starting from `to.block.number` (inclusive),
        // so we pass `first` (the first diverging block) to preserve the common ancestor.
        self.unwind_persistence(*first)?;

        // 2. Unwind Memory
        // Remove `first` and everything after it, mirroring `unwind_persistence(*first)`.
        self.memory.unwind(first.block.number);
        let unwind_duration = unwind_start.elapsed();

        // 3. Store new blocks in In-Memory Buffer
        // Just insert them. They become the new tip.
        for (block, trie_updates, hashed_state) in &block_updates {
            self.memory.insert(
                *block,
                BlockStateDiff {
                    sorted_trie_updates: (**trie_updates).clone(),
                    sorted_post_state: (**hashed_state).clone(),
                },
            );
        }

        let total_duration = start.elapsed();

        #[cfg(feature = "metrics")]
        {
            self.metrics.total_duration_seconds.record(total_duration);
            self.metrics.unwind_duration_seconds.record(unwind_duration);
        }

        info!(
            start_block_number = block_updates.first().map(|(b, _, _)| b.block.number),
            end_block_number = block_updates.last().map(|(b, _, _)| b.block.number),
            ?total_duration,
            ?unwind_duration,
            "Trie updates rewound and buffered successfully",
        );

        // Check if we need to flush (this might happen if the reorg introduced many blocks)
        self.advance_persistence()?;

        Ok(())
    }

    /// Remove account, storage and trie updates from history starting from `to` (inclusive).
    ///
    /// After this call, state up to `to.block.number - 1` is preserved; `to` itself and
    /// all later blocks are removed from both disk and in-memory buffer.
    pub fn unwind_history(&self, to: BlockWithParent) -> Result<(), OpProofsStorageError> {
        info!(target: "live-trie", to_block = to.block.number, "Unwinding history");

        // 1. Unwind Persistence (Disk)
        // `unwind_history` on the store removes `to.block.number..=latest` and sets
        // latest to `to.block.number - 1`, so `to` itself is removed.
        self.unwind_persistence(to)?;

        // 2. Unwind Memory
        // Mirror disk: remove `to` and everything after it.
        self.memory.unwind(to.block.number);

        Ok(())
    }

    /// Returns the (number, hash) of the true tip of the collector.
    fn get_tip(&self) -> Result<NumHash, OpProofsStorageError> {
        let memory_inner = self.memory.inner();
        let numbers = memory_inner.numbers.read();

        // Check memory first
        if let Some((&highest_num, &highest_hash)) = numbers.iter().next_back() {
            return Ok(NumHash::new(highest_num, highest_hash));
        }

        // Fallback to storage
        self.storage
            .provider_ro()?
            .get_latest_block_number()?
            .map(|(n, h)| NumHash::new(n, h))
            .ok_or(OpProofsStorageError::NoBlocksFound)
    }

    /// Returns the block number of the true tip of the collector.
    ///
    /// This resolves to the highest block in the memory buffer, or falls back to
    /// the storage tip if the buffer is empty.
    pub fn get_tip_block_number(&self) -> Result<u64, OpProofsStorageError> {
        let memory_inner = self.memory.inner();
        let numbers = memory_inner.numbers.read();

        // Check memory first
        if let Some(&highest) = numbers.keys().next_back() {
            return Ok(highest);
        }

        // Fallback to storage
        self.storage
            .provider_ro()?
            .get_latest_block_number()?
            .map(|(n, _)| n)
            .ok_or_else(|| OpProofsStorageError::NoBlocksFound)
    }

    /// Blocks the current thread until any in-progress background persistence completes.
    pub fn wait_for_persistence(&self) {
        self.persistence_status.wait_until_idle();
    }

    /// Checks the persistence threshold and triggers persistence if necessary.
    ///
    /// - If buffer >= backpressure: Blocks current thread until persistence frees up space.
    /// - If buffer >= persistence: Triggers background persistence if not already running.
    pub fn advance_persistence(&self) -> Result<(), OpProofsStorageError> {
        let current_size = {
            self.memory.inner().numbers.read().len() as u64
        };

        // 1. Backpressure Check (Blocking)
        // If we are over the limit, we MUST wait for the background task to clear some space.
        if current_size >= self.backpressure_threshold {
            if self.persistence_status.is_running() {
                info!(
                    target: "live-trie",
                    current_size,
                    threshold = self.backpressure_threshold,
                    "Backpressure triggered: Blocking execution until persistence completes"
                );

                self.persistence_status.wait_until_idle();

                info!(target: "live-trie", "Backpressure released: Persistence task completed");
            }
        }

        // 2. Persistence Trigger Check (Async)
        // We re-check the size because if we waited, the memory was pruned.
        let current_size = {
            self.memory.inner().numbers.read().len() as u64
        };

        if current_size >= self.persistence_threshold {
            if self.persistence_status.mark_running() {
                // Snapshot blocks to persist
                let blocks_to_persist = self.get_blocks_to_persist();

                if blocks_to_persist.is_empty() {
                    self.persistence_status.mark_idle();
                    return Ok(());
                }

                info!(
                    target: "live-trie",
                    current_size,
                    count = blocks_to_persist.len(),
                    start_block = blocks_to_persist.first().map(|arc| arc.0.block.number),
                    end_block = blocks_to_persist.last().map(|arc| arc.0.block.number),
                    threshold = self.persistence_threshold,
                    "Persistence threshold reached: Spawning background persistence task"
                );

                // Clone data for the background thread
                let persistence_handle = self.persistence_handle.clone();
                let persistence_status = self.persistence_status.clone();
                let memory = self.memory.clone();

                std::thread::spawn(move || {
                    let result = Self::persist_blocks_background(persistence_handle, blocks_to_persist);

                    match result {
                        Ok(Some(last_persisted)) => {
                            info!(
                                target: "live-trie",
                                block_number = last_persisted,
                                "Background persistence successful, pruning memory"
                            );
                            memory.prune_before(last_persisted + 1);
                        }
                        Ok(None) => {}
                        Err(e) => {
                            error!(target: "live-trie", ?e, "Background persistence failed");
                        }
                    }

                    // Notify completion
                    persistence_status.mark_idle();
                });
            }
        }

        Ok(())
    }

    /// Helper to perform persistence interaction in a background thread.
    fn persist_blocks_background(
        handle: LiveTriePersistenceHandle,
        blocks: Vec<Arc<(BlockWithParent, BlockStateDiff)>>,
    ) -> Result<Option<u64>, OpProofsStorageError> {
        let (tx, rx) = bounded(1);
        handle.save_updates(blocks, tx)?;

        match rx.recv_timeout(Duration::from_secs(DEFAULT_PERSISTENCE_TIMEOUT_SECS)) {
            Ok(res) => Ok(res),
            Err(RecvTimeoutError::Timeout) => {
                Err(OpProofsStorageError::Other("Persistence timeout".into()))
            }
            Err(RecvTimeoutError::Disconnected) => {
                Err(OpProofsStorageError::Other("Persistence service disconnected".into()))
            }
        }
    }

    /// Returns all buffered blocks to persist, ordered from Oldest to Newest.
    ///
    /// Returns `Arc`s to avoid deep-cloning `BlockStateDiff` on the caller thread.
    /// The persistence thread will unwrap or clone as needed.
    fn get_blocks_to_persist(&self) -> Vec<Arc<(BlockWithParent, BlockStateDiff)>> {
        let memory_inner = self.memory.inner();
        let numbers = memory_inner.numbers.read();
        let blocks = memory_inner.blocks.read();

        let mut blocks_to_persist = Vec::with_capacity(numbers.len());

        // BTreeMap is sorted by keys (block numbers), ensuring implicit Oldest -> Newest order.
        for hash in numbers.values() {
            if let Some(state) = blocks.get(hash) {
                blocks_to_persist.push(Arc::clone(state));
            }
        }
        blocks_to_persist
    }

    /// Helper to send unwind command to persistence service and wait for completion.
    fn unwind_persistence(&self, to: BlockWithParent) -> Result<(), OpProofsStorageError> {
        // Wait for any ongoing persistence to finish to avoid race conditions
        if self.persistence_status.is_running() {
            info!(target: "live-trie", "Unwind waiting for background persistence...");
            self.persistence_status.wait_until_idle();
        }

        let (tx, rx) = bounded(1);
        self.persistence_handle.unwind(to, tx)?;

        match rx.recv_timeout(Duration::from_secs(DEFAULT_PERSISTENCE_TIMEOUT_SECS)) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(reason)) => Err(OpProofsStorageError::Other(
                format!("Unwind failed in persistence service: {reason}")
            )),
            Err(RecvTimeoutError::Timeout) => Err(OpProofsStorageError::Other("Unwind timeout".into())),
            Err(RecvTimeoutError::Disconnected) => Err(OpProofsStorageError::Other("Persistence service disconnected".into())),
        }
    }
}