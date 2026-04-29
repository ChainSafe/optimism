//! [`EngineHandle`] — the public, clonable, Send + Sync interface.

use super::{
    engine::Engine,
    tasks::{ExecuteBlockTask, IndexBlockTask, ReorgTask, SyncToTask, UnwindTask},

    EngineAction, DEFAULT_BACKPRESSURE_THRESHOLD, DEFAULT_PERSISTENCE_THRESHOLD,
};
use crate::{OpProofStoragePruner, OpProofsStore};
use super::error::EngineError;
use crossbeam_channel::{bounded, Sender};
use reth_evm::ConfigureEvm;
use reth_primitives_traits::{NodePrimitives, RecoveredBlock};
use reth_provider::{
    BlockHashReader, BlockReader, DatabaseProviderFactory, StateProviderFactory, StateReader,
};
use reth_trie_common::{updates::TrieUpdatesSorted, HashedPostStateSorted};
use std::{panic, sync::Arc, thread};
use tracing::error;

/// A thin, clonable handle used to communicate with the collector engine.
///
/// Every public method sends a [`EngineAction`] to the engine thread and
/// blocks on a one-shot reply channel, preserving the same synchronous API as
/// the old `LiveTrieCollector`.
#[derive(Debug)]
pub struct EngineHandle<Block: reth_primitives_traits::Block> {
    sender: Sender<EngineAction<Block>>,
}

impl<Block: reth_primitives_traits::Block> Clone for EngineHandle<Block> {
    fn clone(&self) -> Self {
        Self { sender: self.sender.clone() }
    }
}

impl<Block: reth_primitives_traits::Block + Send + 'static> EngineHandle<Block> {
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
            + BlockReader<Block = Block>
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
            + BlockReader<Block = Block>
            + Clone
            + 'static,
        Store: OpProofsStore + Clone + 'static,
    {
        let (tx, rx) = bounded(10);
        let engine = Engine::new(evm_config, provider, storage, pruner, rx)
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

    fn send_and_recv(
        &self,
        make_action: impl FnOnce(Sender<Result<(), EngineError>>) -> EngineAction<Block>,
    ) -> Result<(), EngineError> {
        let (reply_tx, reply_rx) = bounded(1);
        self.sender.send(make_action(reply_tx)).map_err(|_| EngineError::EngineDied)?;
        reply_rx.recv().map_err(|_| EngineError::EngineDied)?
    }

    /// Execute a block and store the updates in the in-memory buffer.
    pub fn execute_block(
        &self,
        block: &RecoveredBlock<Block>,
    ) -> Result<(), EngineError>
    where
        Block: Clone,
    {
        self.send_and_recv(|reply| {
            EngineAction::ExecuteBlock(ExecuteBlockTask { block: block.clone(), reply })
        })
    }

    /// Store pre-computed trie updates for a given block.
    pub fn index_block(
        &self,
        block: alloy_eips::eip1898::BlockWithParent,
        sorted_trie_updates: TrieUpdatesSorted,
        sorted_post_state: HashedPostStateSorted,
    ) -> Result<(), EngineError> {
        self.send_and_recv(|reply| {
            EngineAction::IndexBlock(IndexBlockTask {
                block,
                sorted_trie_updates,
                sorted_post_state,
                reply,
            })
        })
    }

    /// Handle a chain reorganisation: unwind then store new blocks.
    pub fn reorg(
        &self,
        block_updates: Vec<(
            alloy_eips::eip1898::BlockWithParent,
            Arc<TrieUpdatesSorted>,
            Arc<HashedPostStateSorted>,
        )>,
    ) -> Result<(), EngineError> {
        self.send_and_recv(|reply| {
            EngineAction::Reorg(ReorgTask { block_updates, reply })
        })
    }

    /// Remove account, storage and trie updates from history starting from `to` (inclusive).
    pub fn unwind(
        &self,
        to: alloy_eips::eip1898::BlockWithParent,
    ) -> Result<(), EngineError> {
        self.send_and_recv(|reply| EngineAction::Unwind(UnwindTask { to, reply }))
    }

    /// Blocks the current thread until any in-progress background persistence completes.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn flush(&self) {
        use super::tasks::FlushTask;
        let (reply_tx, reply_rx) = bounded(1);
        if self.sender.send(EngineAction::Flush(FlushTask { reply: reply_tx })).is_ok() {
            let _ = reply_rx.recv();
        }
    }

    /// Update the sync catch-up target. The engine will execute blocks up to `target`
    /// in its idle time, prioritising any incoming actions over sync work.
    pub fn sync_to(&self, target: u64) -> Result<(), EngineError> {
        self.sender
            .send(EngineAction::SyncTo(SyncToTask { target }))
            .map_err(|_| EngineError::EngineDied)
    }
}
