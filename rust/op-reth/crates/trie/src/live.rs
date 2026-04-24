//! Live trie collector for external proofs storage.
//!
//! The collector runs as an **engine** on a dedicated background thread.  Callers
//! interact with it through [`LiveTrieCollectorHandle`], a thin channel-based
//! handle whose methods mirror the old `LiveTrieCollector` API.
//!
//! Internally the engine owns *all* mutable state (memory buffer, persistence
//! handle, in-flight tracking) and processes [`CollectorAction`] messages one at
//! a time, which structurally enforces the serial-call invariant.

use crate::{
    persistence::LiveTriePersistenceHandle,
    provider::OpProofsStateProviderRef,
    state::LiveTrieState,
    BlockStateDiff, OpProofStoragePruner, OpProofsProviderRO, OpProofsStorageError, OpProofsStore,
};
#[cfg(feature = "metrics")]
use crate::metrics::LiveMetrics;
use alloy_eips::{eip1898::BlockWithParent, NumHash};
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use reth_evm::{execute::Executor, ConfigureEvm};
use reth_primitives_traits::{AlloyBlockHeader, BlockTy, NodePrimitives, RecoveredBlock};
use reth_provider::{
    BlockHashReader, DatabaseProviderFactory, HashedPostStateProvider, StateProviderFactory,
    StateReader, StateRootProvider,
};
use reth_revm::database::StateProviderDatabase;
use reth_trie_common::{updates::TrieUpdatesSorted, HashedPostStateSorted};
use std::{
    panic,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
use tracing::{debug, error, info};

/// Default number of blocks to keep in memory before persisting.
pub const DEFAULT_PERSISTENCE_THRESHOLD: u64 = 5;

/// Default number of blocks where we block execution to allow persistence to catch up.
pub const DEFAULT_BACKPRESSURE_THRESHOLD: u64 = 10;

/// Default timeout for waiting on a persistence save/unwind operation (in seconds).
pub const DEFAULT_PERSISTENCE_TIMEOUT_SECS: u64 = 60;

// ---------------------------------------------------------------------------
// CollectorAction – messages sent from the handle to the engine
// ---------------------------------------------------------------------------

/// Messages sent from [`LiveTrieCollectorHandle`] to the engine thread.
enum CollectorAction<Block: reth_primitives_traits::Block> {
    /// Execute a block and store the resulting trie diff in the memory buffer.
    ExecuteAndStore {
        block: RecoveredBlock<Block>,
        reply: Sender<Result<(), OpProofsStorageError>>,
    },
    /// Store pre-computed trie updates for a block.
    StoreBlockUpdates {
        block: BlockWithParent,
        sorted_trie_updates: TrieUpdatesSorted,
        sorted_post_state: HashedPostStateSorted,
        reply: Sender<Result<(), OpProofsStorageError>>,
    },
    /// Unwind then store new blocks (reorg handling).
    UnwindAndStore {
        block_updates: Vec<(BlockWithParent, Arc<TrieUpdatesSorted>, Arc<HashedPostStateSorted>)>,
        reply: Sender<Result<(), OpProofsStorageError>>,
    },
    /// Unwind history to a given block (inclusive removal).
    UnwindHistory {
        to: BlockWithParent,
        reply: Sender<Result<(), OpProofsStorageError>>,
    },
    /// Block the caller until any in-flight persistence completes.
    WaitForPersistence {
        reply: Sender<()>,
    },
    /// Query the tip block number.
    GetTipBlockNumber {
        reply: Sender<Result<u64, OpProofsStorageError>>,
    },
}

// ---------------------------------------------------------------------------
// LiveTrieCollectorHandle – the public, clonable, Send + Sync interface
// ---------------------------------------------------------------------------

/// A thin, clonable handle used to communicate with the collector engine.
///
/// Every public method sends a [`CollectorAction`] to the engine thread and
/// blocks on a one-shot reply channel, preserving the same synchronous API as
/// the old `LiveTrieCollector`.
#[derive(Debug)]
pub struct LiveTrieCollectorHandle<Block: reth_primitives_traits::Block> {
    sender: Sender<CollectorAction<Block>>,
}

impl<Block: reth_primitives_traits::Block> Clone for LiveTrieCollectorHandle<Block> {
    fn clone(&self) -> Self {
        Self { sender: self.sender.clone() }
    }
}

impl<Block: reth_primitives_traits::Block + Send + 'static> LiveTrieCollectorHandle<Block> {
    /// Spawn the collector engine on a new thread and return a handle.
    pub fn spawn<Evm, Provider, Store>(
        evm_config: Evm,
        provider: Provider,
        storage: Store,
        pruner: OpProofStoragePruner<Store, Provider>,
    ) -> Self
    where
        Evm: ConfigureEvm<Primitives: NodePrimitives<Block = Block>> + 'static,
        Provider: BlockHashReader
            + StateReader
            + DatabaseProviderFactory
            + StateProviderFactory
            + Clone
            + 'static,
        Store: OpProofsStore + Clone + 'static,
    {
        Self::spawn_with_thresholds(
            evm_config,
            provider,
            storage,
            pruner,
            DEFAULT_PERSISTENCE_THRESHOLD,
            DEFAULT_BACKPRESSURE_THRESHOLD,
        )
    }

    /// Spawn the collector engine with custom thresholds.
    pub fn spawn_with_thresholds<Evm, Provider, Store>(
        evm_config: Evm,
        provider: Provider,
        storage: Store,
        pruner: OpProofStoragePruner<Store, Provider>,
        persistence_threshold: u64,
        backpressure_threshold: u64,
    ) -> Self
    where
        Evm: ConfigureEvm<Primitives: NodePrimitives<Block = Block>> + 'static,
        Provider: BlockHashReader
            + StateReader
            + DatabaseProviderFactory
            + StateProviderFactory
            + Clone
            + 'static,
        Store: OpProofsStore + Clone + 'static,
    {
        let (tx, rx) = bounded(4);

        let engine = LiveTrieCollectorEngine::new(evm_config, provider, storage, pruner, rx)
            .with_persistence_threshold(persistence_threshold)
            .with_backpressure_threshold(backpressure_threshold);

        thread::Builder::new()
            .name("live-trie-collector".into())
            .spawn(move || {
                if let Err(panic) =
                    panic::catch_unwind(panic::AssertUnwindSafe(|| engine.run()))
                {
                    let msg = panic
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
                        .unwrap_or("unknown");
                    error!(target: "live-trie::engine", %msg, "Collector engine panicked");
                }
            })
            .expect("failed to spawn live-trie-collector thread");

        Self { sender: tx }
    }

    // -- Helper: send action, wait for reply --------------------------------

    fn send_and_recv<R>(
        &self,
        make_action: impl FnOnce(Sender<R>) -> CollectorAction<Block>,
    ) -> Result<R, OpProofsStorageError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.sender
            .send(make_action(reply_tx))
            .map_err(|_| OpProofsStorageError::Other("Collector engine died".into()))?;
        reply_rx
            .recv()
            .map_err(|_| OpProofsStorageError::Other("Collector engine died".into()))
    }

    // -- Public API ---------------------------------------------------------

    /// Execute a block and store the updates in the in-memory buffer.
    pub fn execute_and_store_block_updates(
        &self,
        block: &RecoveredBlock<Block>,
    ) -> Result<(), OpProofsStorageError>
    where
        Block: Clone,
    {
        self.send_and_recv(|reply| CollectorAction::ExecuteAndStore {
            block: block.clone(),
            reply,
        })?
    }

    /// Store pre-computed trie updates for a given block.
    pub fn store_block_updates(
        &self,
        block: BlockWithParent,
        sorted_trie_updates: TrieUpdatesSorted,
        sorted_post_state: HashedPostStateSorted,
    ) -> Result<(), OpProofsStorageError> {
        self.send_and_recv(|reply| CollectorAction::StoreBlockUpdates {
            block,
            sorted_trie_updates,
            sorted_post_state,
            reply,
        })?
    }

    /// Handle a chain reorganisation: unwind then store new blocks.
    pub fn unwind_and_store_block_updates(
        &self,
        block_updates: Vec<(
            BlockWithParent,
            Arc<TrieUpdatesSorted>,
            Arc<HashedPostStateSorted>,
        )>,
    ) -> Result<(), OpProofsStorageError> {
        self.send_and_recv(|reply| CollectorAction::UnwindAndStore { block_updates, reply })?
    }

    /// Remove account, storage and trie updates from history starting from `to` (inclusive).
    pub fn unwind_history(&self, to: BlockWithParent) -> Result<(), OpProofsStorageError> {
        self.send_and_recv(|reply| CollectorAction::UnwindHistory { to, reply })?
    }

    /// Blocks the current thread until any in-progress background persistence completes.
    pub fn wait_for_persistence(&self) {
        let (reply_tx, reply_rx) = bounded(1);
        if self.sender.send(CollectorAction::WaitForPersistence { reply: reply_tx }).is_ok() {
            let _ = reply_rx.recv();
        }
    }

    /// Returns the block number of the true tip of the collector.
    pub fn get_tip_block_number(&self) -> Result<u64, OpProofsStorageError> {
        self.send_and_recv(|reply| CollectorAction::GetTipBlockNumber { reply })?
    }
}

// ---------------------------------------------------------------------------
// LiveTrieCollectorEngine – the single-threaded engine that owns all state
// ---------------------------------------------------------------------------

/// The engine that runs on a dedicated thread, processing [`CollectorAction`]
/// messages sequentially.
struct LiveTrieCollectorEngine<Evm, Provider, Store>
where
    Evm: ConfigureEvm,
    Provider: StateReader + DatabaseProviderFactory + StateProviderFactory,
{
    evm_config: Evm,
    provider: Provider,
    storage: Store,
    memory: LiveTrieState,

    persistence_threshold: u64,
    backpressure_threshold: u64,
    persistence_handle: LiveTriePersistenceHandle,

    /// Whether a persistence save is currently in-flight.
    persist_in_flight: bool,
    /// Reply channel for the in-flight persistence save (if any).
    persist_reply_rx: Option<Receiver<Option<u64>>>,

    incoming: Receiver<CollectorAction<BlockTy<Evm::Primitives>>>,

    #[cfg(feature = "metrics")]
    metrics: LiveMetrics,
}

impl<Evm, Provider, Store> LiveTrieCollectorEngine<Evm, Provider, Store>
where
    Evm: ConfigureEvm,
    Provider: BlockHashReader
        + StateReader
        + DatabaseProviderFactory
        + StateProviderFactory
        + Clone
        + 'static,
    Store: OpProofsStore + Clone + 'static,
{
    fn new(
        evm_config: Evm,
        provider: Provider,
        storage: Store,
        pruner: OpProofStoragePruner<Store, Provider>,
        incoming: Receiver<CollectorAction<BlockTy<Evm::Primitives>>>,
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

            persist_in_flight: false,
            persist_reply_rx: None,

            incoming,

            #[cfg(feature = "metrics")]
            metrics: LiveMetrics::new_with_labels(&[] as &[(&str, &str)]),
        }
    }

    fn with_persistence_threshold(mut self, threshold: u64) -> Self {
        self.persistence_threshold = threshold;
        self
    }

    fn with_backpressure_threshold(mut self, threshold: u64) -> Self {
        self.backpressure_threshold = threshold;
        self
    }

    // -----------------------------------------------------------------------
    // Main event loop
    // -----------------------------------------------------------------------

    fn run(mut self) {
        debug_assert!(
            self.persistence_threshold < self.backpressure_threshold,
            "backpressure_threshold ({}) must be greater than persistence_threshold ({})",
            self.backpressure_threshold,
            self.persistence_threshold,
        );
        debug!(target: "live-trie::engine", "Collector engine started");

        while let Ok(action) = self.incoming.recv() {
            match action {
                CollectorAction::ExecuteAndStore { block, reply } => {
                    let result = self.do_execute_and_store(&block);
                    let _ = reply.send(result);
                }
                CollectorAction::StoreBlockUpdates {
                    block,
                    sorted_trie_updates,
                    sorted_post_state,
                    reply,
                } => {
                    let result =
                        self.do_store_block_updates(block, sorted_trie_updates, sorted_post_state);
                    let _ = reply.send(result);
                }
                CollectorAction::UnwindAndStore { block_updates, reply } => {
                    let result = self.do_unwind_and_store(block_updates);
                    let _ = reply.send(result);
                }
                CollectorAction::UnwindHistory { to, reply } => {
                    let result = self.do_unwind_history(to);
                    let _ = reply.send(result);
                }
                CollectorAction::WaitForPersistence { reply } => {
                    self.wait_for_persist_result();
                    let _ = reply.send(());
                }
                CollectorAction::GetTipBlockNumber { reply } => {
                    let _ = reply.send(self.get_tip_block_number());
                }
            }
        }

        // Channel closed — handle is dropped, shut down gracefully.
        debug!(target: "live-trie::engine", "Collector engine shutting down, draining in-flight persist");
        self.wait_for_persist_result();
        debug!(target: "live-trie::engine", "Collector engine stopped");
    }

    // -----------------------------------------------------------------------
    // Persistence management
    // -----------------------------------------------------------------------

    /// Non-blocking: if a persist is in-flight and has completed, collect its
    /// result and prune memory.
    fn try_collect_persist_result(&mut self) {
        if !self.persist_in_flight {
            return;
        }

        let Some(rx) = self.persist_reply_rx.take() else { return };

        match rx.try_recv() {
            Ok(Some(last_persisted)) => {
                info!(
                    target: "live-trie::engine",
                    block_number = last_persisted,
                    "Background persistence completed, pruning memory"
                );
                self.memory.prune_before(last_persisted + 1);
                self.persist_in_flight = false;
            }
            Ok(None) => {
                // Persist returned None (empty batch?), mark done.
                self.persist_in_flight = false;
            }
            Err(crossbeam_channel::TryRecvError::Empty) => {
                // Still running — put the receiver back and leave in-flight.
                self.persist_reply_rx = Some(rx);
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                error!(target: "live-trie::engine", "Persistence service disconnected while in-flight");
                self.persist_in_flight = false;
            }
        }
    }

    /// Blocking: wait for the in-flight persist to finish and collect its result.
    fn wait_for_persist_result(&mut self) {
        if !self.persist_in_flight {
            return;
        }

        if let Some(rx) = self.persist_reply_rx.take() {
            match rx.recv_timeout(Duration::from_secs(DEFAULT_PERSISTENCE_TIMEOUT_SECS)) {
                Ok(Some(last_persisted)) => {
                    info!(
                        target: "live-trie::engine",
                        block_number = last_persisted,
                        "Persistence completed (waited), pruning memory"
                    );
                    self.memory.prune_before(last_persisted + 1);
                }
                Ok(None) => {}
                Err(RecvTimeoutError::Timeout) => {
                    error!(target: "live-trie::engine", "Persistence timeout while waiting");
                }
                Err(RecvTimeoutError::Disconnected) => {
                    error!(target: "live-trie::engine", "Persistence service disconnected while waiting");
                }
            }
        }
        self.persist_in_flight = false;
    }

    /// Check the buffer size and optionally trigger a non-blocking persist, or
    /// block if backpressure is required.
    fn advance_persistence(&mut self) -> Result<(), OpProofsStorageError> {
        // Collect any completed persist result before reading the buffer size, so
        // threshold decisions are based on the post-prune state rather than stale counts.
        self.try_collect_persist_result();

        let current_size = self.memory.inner().numbers.read().len() as u64;

        // 1. Backpressure: if buffer is too large and persist is in-flight, wait.
        if current_size >= self.backpressure_threshold && self.persist_in_flight {
            info!(
                target: "live-trie::engine",
                current_size,
                threshold = self.backpressure_threshold,
                "Backpressure triggered: waiting for persistence to complete"
            );
            self.wait_for_persist_result();
            info!(target: "live-trie::engine", "Backpressure released");
        }

        // Re-check after possible wait.
        let current_size = self.memory.inner().numbers.read().len() as u64;

        // 2. Trigger a new persist if threshold met and nothing in-flight.
        if current_size >= self.persistence_threshold && !self.persist_in_flight {
            let blocks = self.get_blocks_to_persist();
            if blocks.is_empty() {
                return Ok(());
            }

            info!(
                target: "live-trie::engine",
                current_size,
                count = blocks.len(),
                start_block = blocks.first().map(|arc| arc.0.block.number),
                end_block = blocks.last().map(|arc| arc.0.block.number),
                threshold = self.persistence_threshold,
                "Persistence threshold reached: sending to persistence service"
            );

            let (tx, rx) = bounded(1);
            self.persistence_handle.save_updates(blocks, tx)?;
            self.persist_in_flight = true;
            self.persist_reply_rx = Some(rx);
        }

        Ok(())
    }

    /// Wait for in-flight persist then send unwind to persistence service and
    /// wait for its completion.
    fn unwind_persistence(&mut self, to: BlockWithParent) -> Result<(), OpProofsStorageError> {
        // Must wait for any in-flight persist to avoid race.
        if self.persist_in_flight {
            info!(target: "live-trie::engine", "Unwind waiting for in-flight persistence...");
            self.wait_for_persist_result();
        }

        let (tx, rx) = bounded(1);
        self.persistence_handle.unwind(to, tx)?;

        match rx.recv_timeout(Duration::from_secs(DEFAULT_PERSISTENCE_TIMEOUT_SECS)) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(reason)) => Err(OpProofsStorageError::Other(format!(
                "Unwind failed in persistence service: {reason}"
            ))),
            Err(RecvTimeoutError::Timeout) => {
                Err(OpProofsStorageError::Other("Unwind timeout".into()))
            }
            Err(RecvTimeoutError::Disconnected) => {
                Err(OpProofsStorageError::Other("Persistence service disconnected".into()))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Block operations
    // -----------------------------------------------------------------------

    fn do_execute_and_store(
        &mut self,
        block: &RecoveredBlock<BlockTy<Evm::Primitives>>,
    ) -> Result<(), OpProofsStorageError> {
        let start = Instant::now();

        let tip = self.get_tip()?;
        let parent_block_number = block.number().saturating_sub(1);

        if block.parent_hash() != tip.hash {
            return Err(OpProofsStorageError::OutOfOrder {
                block_number: block.number(),
                parent_block_hash: block.parent_hash(),
                latest_block_hash: tip.hash,
            });
        }

        let block_ref =
            BlockWithParent::new(block.parent_hash(), NumHash::new(block.number(), block.hash()));

        // Scope the immutable borrows of `self.storage` / `self.memory` so they
        // are dropped before we call `self.advance_persistence()` below.
        let (sorted_trie_updates, sorted_post_state, execution_duration, state_root_duration) = {
            let inner_provider = OpProofsStateProviderRef::new(
                self.provider.state_by_block_hash(block.parent_hash())?,
                self.storage.provider_ro()?,
                parent_block_number,
            );

            let state_provider = self.memory.state_provider(block.parent_hash(), inner_provider);

            let db = StateProviderDatabase::new(&state_provider);
            let block_executor = self.evm_config.batch_executor(db);
            let execution_result = block_executor.execute(block)?;
            let execution_duration = start.elapsed();

            let hashed_state = state_provider.hashed_post_state(&execution_result.state);
            let (state_root, trie_updates) =
                state_provider.state_root_with_updates(hashed_state.clone())?;
            let state_root_duration = start.elapsed() - execution_duration;

            if state_root != block.state_root() {
                return Err(OpProofsStorageError::StateRootMismatch {
                    block_number: block.number(),
                    current_state_hash: state_root,
                    expected_state_hash: block.state_root(),
                });
            }

            (
                trie_updates.into_sorted(),
                hashed_state.into_sorted(),
                execution_duration,
                state_root_duration,
            )
        };

        self.memory.insert(
            block_ref,
            BlockStateDiff { sorted_trie_updates, sorted_post_state },
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

        self.advance_persistence()?;
        Ok(())
    }

    fn do_store_block_updates(
        &mut self,
        block: BlockWithParent,
        sorted_trie_updates: TrieUpdatesSorted,
        sorted_post_state: HashedPostStateSorted,
    ) -> Result<(), OpProofsStorageError> {
        let start = Instant::now();

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

        self.advance_persistence()?;
        Ok(())
    }

    fn do_unwind_and_store(
        &mut self,
        block_updates: Vec<(
            BlockWithParent,
            Arc<TrieUpdatesSorted>,
            Arc<HashedPostStateSorted>,
        )>,
    ) -> Result<(), OpProofsStorageError> {
        if block_updates.is_empty() {
            return Ok(());
        }

        let start = Instant::now();
        let first = &block_updates[0].0;
        let common_ancestor_number = first.block.number.saturating_sub(1);

        info!(
            target: "live-trie::engine",
            reorg_depth = block_updates.len(),
            common_ancestor = common_ancestor_number,
            "Handling reorg: unwinding and buffering new path"
        );

        let unwind_start = Instant::now();
        self.unwind_persistence(*first)?;
        self.memory.unwind(first.block.number);
        let unwind_duration = unwind_start.elapsed();

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

        self.advance_persistence()?;
        Ok(())
    }

    fn do_unwind_history(&mut self, to: BlockWithParent) -> Result<(), OpProofsStorageError> {
        info!(target: "live-trie::engine", to_block = to.block.number, "Unwinding history");
        self.unwind_persistence(to)?;
        self.memory.unwind(to.block.number);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn get_tip(&self) -> Result<NumHash, OpProofsStorageError> {
        let memory_inner = self.memory.inner();
        let numbers = memory_inner.numbers.read();

        if let Some((&highest_num, &highest_hash)) = numbers.iter().next_back() {
            return Ok(NumHash::new(highest_num, highest_hash));
        }

        self.storage
            .provider_ro()?
            .get_latest_block_number()?
            .map(|(n, h)| NumHash::new(n, h))
            .ok_or(OpProofsStorageError::NoBlocksFound)
    }

    fn get_tip_block_number(&self) -> Result<u64, OpProofsStorageError> {
        self.get_tip().map(|tip| tip.number)
    }

    /// Returns all buffered blocks to persist, ordered oldest to newest.
    fn get_blocks_to_persist(&self) -> Vec<Arc<(BlockWithParent, BlockStateDiff)>> {
        let memory_inner = self.memory.inner();
        let numbers = memory_inner.numbers.read();
        let blocks = memory_inner.blocks.read();

        let mut out = Vec::with_capacity(numbers.len());
        for hash in numbers.values() {
            if let Some(state) = blocks.get(hash) {
                out.push(Arc::clone(state));
            }
        }
        out
    }
}
