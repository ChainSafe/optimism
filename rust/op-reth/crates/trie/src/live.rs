//! Live trie collector for external proofs storage.

use crate::{
    api::OperationDurations,
    provider::OpProofsStateProviderRef, state::LiveTrieState, BlockStateDiff, OpProofsStorage,
    OpProofsStorageError, OpProofsStore, persistence::LiveTriePersistenceHandle,
};
use alloy_primitives::B256;
use alloy_eips::{eip1898::BlockWithParent, BlockNumHash, NumHash};
use crossbeam_channel::{bounded, RecvTimeoutError};
use reth_evm::{execute::Executor, ConfigureEvm};
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, RecoveredBlock};
use reth_provider::{
    DatabaseProviderFactory, HashedPostStateProvider, StateProviderFactory, StateReader,
    StateRootProvider,
};
use reth_revm::database::StateProviderDatabase;
use reth_trie_common::{updates::TrieUpdatesSorted, HashedPostStateSorted};
use std::{sync::{Arc, Mutex, Condvar}, time::{Duration, Instant}};
use tracing::{error, info};

/// Default number of blocks to keep in memory before persisting.
pub const DEFAULT_PERSISTENCE_THRESHOLD: u64 = 5;

/// Default number of blocks where we block execution to allow persistence to catch up.
pub const DEFAULT_BACKPRESSURE_THRESHOLD: u64 = 10;

/// Live trie collector for external proofs storage.
#[derive(Debug)]
pub struct LiveTrieCollector<Evm, Provider, PreimageStore>
where
    Evm: ConfigureEvm,
    Provider: StateReader + DatabaseProviderFactory + StateProviderFactory,
{
    evm_config: Evm,
    provider: Provider,
    storage: Arc<OpProofsStorage<PreimageStore>>,
    memory: LiveTrieState,

    /// Number of blocks to keep in memory before persisting.
    persistence_threshold: u64,
    /// Number of blocks to keep in memory limit (backpressure).
    backpressure_threshold: u64,
    persistence_handle: LiveTriePersistenceHandle,
    /// Tracks if a persistence task is currently running.
    /// (is_persisting, condvar)
    persistence_status: Arc<(Mutex<bool>, Condvar)>,
}

impl<Evm, Provider, Store> LiveTrieCollector<Evm, Provider, Store>
where
    Evm: ConfigureEvm,
    Provider: StateReader + DatabaseProviderFactory + StateProviderFactory,
    Store: OpProofsStore + Clone + 'static,
{
    /// Create a new live trie collector.
    pub fn new(
        evm_config: Evm,
        provider: Provider,
        storage: OpProofsStorage<Store>,
    ) -> Self {
        let persistence_handle = LiveTriePersistenceHandle::spawn(storage.clone());
        Self {
            evm_config,
            provider,
            storage: Arc::new(storage),
            memory: LiveTrieState::new(),

            persistence_threshold: DEFAULT_PERSISTENCE_THRESHOLD,
            backpressure_threshold: DEFAULT_BACKPRESSURE_THRESHOLD,
            persistence_handle,
            persistence_status: Arc::new((Mutex::new(false), Condvar::new())),
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
        let mut operation_durations = OperationDurations::default();
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
            self.storage.as_ref(),
            parent_block_number,
        );

        // 2. Wrap it with memory overlay using LiveTrieState
        // This gathers all buffered blocks required to build the state on top of disk
        let state_provider = self.memory.state_provider(block.parent_hash(), inner_provider);

        // 3. Execute block
        let db = StateProviderDatabase::new(&state_provider);
        let block_executor = self.evm_config.batch_executor(db);

        let execution_result = block_executor.execute(&(*block).clone())?;

        operation_durations.execution_duration_seconds = start.elapsed();

        // 4. Calculate state root
        let hashed_state = state_provider.hashed_post_state(&execution_result.state);
        let (state_root, trie_updates) =
            state_provider.state_root_with_updates(hashed_state.clone())?;

        operation_durations.state_root_duration_seconds =
            start.elapsed() - operation_durations.execution_duration_seconds;

        // 5. Verify root
        if state_root != block.state_root() {
            return Err(OpProofsStorageError::StateRootMismatch {
                block_number: block.number(),
                current_state_hash: state_root,
                expected_state_hash: block.state_root(),
            });
        }

        operation_durations.state_root_duration_seconds =
            start.elapsed() - operation_durations.execution_duration_seconds;

        // 6. Store Diff to Memory
        self.memory.insert(
            block_ref,
            BlockStateDiff {
                sorted_trie_updates: trie_updates.into_sorted(),
                sorted_post_state: hashed_state.into_sorted(),
            },
        );

        operation_durations.total_duration_seconds = start.elapsed();
        operation_durations.write_duration_seconds = operation_durations.total_duration_seconds -
            operation_durations.state_root_duration_seconds -
            operation_durations.execution_duration_seconds;

        #[cfg(feature = "metrics")]
        {
            let block_metrics = self.storage.metrics().block_metrics();
            block_metrics.record_operation_durations(&operation_durations);
            // block_metrics.increment_write_counts(&update_result);
        }

        info!(
            block_number = block.number(),
            ?operation_durations,
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
        let mut operation_durations = OperationDurations::default();

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

        let write_duration = start.elapsed();
        operation_durations.total_duration_seconds = write_duration;
        operation_durations.write_duration_seconds = write_duration;

        #[cfg(feature = "metrics")]
        {
            let block_metrics = self.storage.metrics().block_metrics();
            block_metrics.record_operation_durations(&operation_durations);
            // block_metrics.increment_write_counts(&storage_result);
        }

        info!(
            block_number = block.block.number,
            ?operation_durations,
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
        let mut operation_durations = OperationDurations::default();
        let first = &block_updates[0].0;
         let latest_common_block = BlockWithParent {
            block: BlockNumHash::new(first.block.number.saturating_sub(1), first.parent),
            // todo: pass the actual parent hash of the common ancestor block here instead of assuming it's zero
            parent: B256::ZERO,
        };

        info!(
            target: "live-trie",
            reorg_depth = block_updates.len(),
            common_ancestor = latest_common_block.block.number,
            "Handling reorg: unwinding and buffering new path"
        );

        // 1. Unwind Persistence (Disk)
        // We must perform this to ensure the disk state is valid.
        // We use a dedicated helper that talks to the service.
        self.unwind_persistence(latest_common_block)?;

        // 2. Unwind Memory
        // Remove everything strictly after the common ancestor.
        self.memory.prune_after(latest_common_block.block.number);

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

        let write_duration = start.elapsed();
        operation_durations.total_duration_seconds = write_duration;
        operation_durations.write_duration_seconds = write_duration;

        #[cfg(feature = "metrics")]
        {
            let block_metrics = self.storage.metrics().block_metrics();
            block_metrics.record_operation_durations(&operation_durations);
        }

        info!(
            start_block_number = block_updates.first().map(|(b, _, _)| b.block.number),
            end_block_number = block_updates.last().map(|(b, _, _)| b.block.number),
            ?operation_durations,
            "Trie updates rewound and buffered successfully",
        );

        // Check if we need to flush (this might happen if the reorg introduced many blocks)
        self.advance_persistence()?;

        Ok(())
    }

    /// Remove account, storage and trie updates from historical storage for all blocks from
    /// the specified block (inclusive).
    ///
    /// This keeps state up to `to` (inclusive) and invalidates everything after it.
    pub fn unwind_history(&self, to: BlockWithParent) -> Result<(), OpProofsStorageError> {
        info!(target: "live-trie", to_block = to.block.number, "Unwinding history");

        // 1. Unwind Persistence (Disk)
        self.unwind_persistence(to)?;

        // 2. Unwind Memory
        // Remove everything strictly after `to`
        self.memory.prune_after(to.block.number);

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
        self.storage.get_latest_block_number()?
            .map(|(n, _)| n)
            .ok_or_else(|| OpProofsStorageError::NoBlocksFound)
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
            let (lock, cvar) = &*self.persistence_status;
            let mut is_persisting = lock.lock().map_err(|_| OpProofsStorageError::Other("Mutex poisoned".into()))?;

            if *is_persisting {
                info!(
                    target: "live-trie",
                    current_size,
                    threshold = self.backpressure_threshold,
                    "Backpressure triggered: Blocking execution until persistence completes"
                );

                // Wait while persistence is active.
                while *is_persisting {
                    is_persisting = cvar.wait(is_persisting).map_err(|_| OpProofsStorageError::Other("Condvar poisoned".into()))?;
                }

                info!(target: "live-trie", "Backpressure released: Persistence task completed");
            }
        }

        // 2. Persistence Trigger Check (Async)
        // We re-check the size because if we waited, the memory was pruned.
        let current_size = {
            self.memory.inner().numbers.read().len() as u64
        };

        if current_size >= self.persistence_threshold {
            let (lock, _) = &*self.persistence_status;
            let mut is_persisting = lock.lock().map_err(|_| OpProofsStorageError::Other("Mutex poisoned".into()))?;

            if !*is_persisting {
                // Snapshot blocks to persist
                let blocks_to_persist = self.get_blocks_to_persist();

                if blocks_to_persist.is_empty() {
                    return Ok(());
                }

                info!(
                    target: "live-trie",
                    current_size,
                    count = blocks_to_persist.len(),
                    start_block = blocks_to_persist.first().map(|(b, _)| b.block.number),
                    end_block = blocks_to_persist.last().map(|(b, _)| b.block.number),
                    threshold = self.persistence_threshold,
                    "Persistence threshold reached: Spawning background persistence task"
                );

                *is_persisting = true;

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
                    let (lock, cvar) = &*persistence_status;
                    let mut running = lock.lock().unwrap();
                    *running = false;
                    cvar.notify_all();
                });
            }
        }

        Ok(())
    }

    /// Helper to perform persistence interaction in a background thread.
    fn persist_blocks_background(
        handle: LiveTriePersistenceHandle,
        blocks: Vec<(BlockWithParent, BlockStateDiff)>,
    ) -> Result<Option<u64>, OpProofsStorageError> {
        let (tx, rx) = bounded(1);
        handle.save_updates(blocks, tx);

        match rx.recv_timeout(Duration::from_secs(300)) {
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
    fn get_blocks_to_persist(&self) -> Vec<(BlockWithParent, BlockStateDiff)> {
        let memory_inner = self.memory.inner();
        let numbers = memory_inner.numbers.read();
        let blocks = memory_inner.blocks.read();

        let mut blocks_to_persist = Vec::with_capacity(numbers.len());

        // BTreeMap is sorted by keys (block numbers), ensuring implicit Oldest -> Newest order.
        for hash in numbers.values() {
            if let Some(state) = blocks.get(hash) {
                blocks_to_persist.push((state.0.clone(), state.1.clone()));
            }
        }
        blocks_to_persist
    }

    /// Helper to send unwind command to persistence service and wait for completion.
    fn unwind_persistence(&self, to: BlockWithParent) -> Result<(), OpProofsStorageError> {
        // Wait for any ongoing persistence to finish to avoid race conditions
        {
             let (lock, cvar) = &*self.persistence_status;
             let mut is_persisting = lock.lock().map_err(|_| OpProofsStorageError::Other("Mutex poisoned".into()))?;
             while *is_persisting {
                 info!(target: "live-trie", "Unwind waiting for background persistence...");
                 is_persisting = cvar.wait(is_persisting).map_err(|_| OpProofsStorageError::Other("Condvar poisoned".into()))?;
             }
        }

        let (tx, rx) = bounded(1);
        self.persistence_handle.unwind(to, tx);

        match rx.recv_timeout(Duration::from_secs(60)) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(())) => Err(OpProofsStorageError::Other("Unwind failed in persistence service".into())),
            Err(RecvTimeoutError::Timeout) => Err(OpProofsStorageError::Other("Unwind timeout".into())),
            Err(RecvTimeoutError::Disconnected) => Err(OpProofsStorageError::Other("Persistence service disconnected".into())),
        }
    }
}
