//! Persistence implementation for external proof

use crate::{
    api::{OpProofsProviderRw, WriteCounts},
    prune::OpProofStoragePruner,
    BlockStateDiff, OpProofsStore, OpProofsStorage,
};
use alloy_eips::eip1898::BlockWithParent;
use reth_provider::BlockHashReader;
use crossbeam_channel::{Receiver, Sender};
use std::{thread, time::Instant};
use tracing::{debug, error, info};

/// Messages sent to the persistence service.
#[derive(Debug)]
pub enum LiveTriePersistenceAction {
    /// Save a batch of trie updates to storage.
    ///
    /// Contains:
    /// 1. The list of blocks and their diffs (ordered Oldest -> Newest).
    /// 2. A response channel to return the highest block number persisted (for pruning).
    SaveUpdates(Vec<(BlockWithParent, BlockStateDiff)>, Sender<Option<u64>>),
    /// Unwind history to the specified block (inclusive).
    /// All history strictly after this block is removed.
    Unwind(BlockWithParent, Sender<Result<(), ()>>),
}

/// A handle to communicate with the Live Trie persistence service.
#[derive(Debug, Clone)]
pub struct LiveTriePersistenceHandle {
    sender: Sender<LiveTriePersistenceAction>,
}

impl LiveTriePersistenceHandle {
    /// Create a new handle.
    pub fn new(sender: Sender<LiveTriePersistenceAction>) -> Self {
        Self { sender }
    }

    /// Spawn the service in a new thread and return a handle.
    pub fn spawn<H, S>(pruner: OpProofStoragePruner<S, H>, storage: OpProofsStorage<S>) -> Self
    where
        S: OpProofsStore + Send + Sync + 'static,
        H: BlockHashReader + Send + Sync + 'static,
    {
        let (tx, rx) = crossbeam_channel::unbounded();
        let service = LiveTriePersistenceService::new(pruner, storage, rx);

        thread::Builder::new()
            .name("Live Trie Persistence".into())
            .spawn(move || service.run())
            .expect("failed to spawn live trie persistence thread");

        Self::new(tx)
    }

    /// Send a save request.
    pub fn save_updates(
        &self,
        updates: Vec<(BlockWithParent, BlockStateDiff)>,
        response_tx: Sender<Option<u64>>,
    ) {
        let _ = self.sender.send(LiveTriePersistenceAction::SaveUpdates(updates, response_tx));
    }

    /// Send an unwind request.
    pub fn unwind(
        &self,
        to: BlockWithParent,
        response_tx: Sender<Result<(), ()>>,
    ) {
        let _ = self.sender.send(LiveTriePersistenceAction::Unwind(to, response_tx));
    }
}

/// Service that runs in a background thread to persist trie updates.
#[derive(Debug)]
pub struct LiveTriePersistenceService<H, S> {
    /// Pruner that also owns the storage backend and block hash reader.
    pruner: OpProofStoragePruner<S, H>,
    storage: OpProofsStorage<S>,
    incoming: Receiver<LiveTriePersistenceAction>,
}

impl<H: BlockHashReader, S: OpProofsStore> LiveTriePersistenceService<H, S> {
    /// Create a new persistence service.
    pub fn new(pruner: OpProofStoragePruner<S, H>, storage: OpProofsStorage<S>, incoming: Receiver<LiveTriePersistenceAction>) -> Self {
        Self { pruner, storage, incoming }
    }

    /// Main loop for the service.
    /// Listens for incoming actions and processes them sequentially.
    pub fn run(self) {
        debug!(target: "live-trie::persistence", "Service started");

        while let Ok(action) = self.incoming.recv() {
            match action {
                LiveTriePersistenceAction::Unwind(to, reply_tx) => {
                    self.on_unwind(to, reply_tx);
                }
                LiveTriePersistenceAction::SaveUpdates(updates, reply_tx) => {
                    self.on_save_updates(updates, reply_tx);
                }
            }
        }
        debug!(target: "live-trie::persistence", "Service shutting down");
    }

    fn on_save_updates(
        &self,
        updates: Vec<(BlockWithParent, BlockStateDiff)>,
        reply_tx: Sender<Option<u64>>,
    ) {
        if updates.is_empty() {
            let _ = reply_tx.send(None);
            return;
        }

        let start = Instant::now();
        let count = updates.len();
        let first = updates.first().map(|(b, _)| b.block.number);
        let last = updates.last().map(|(b, _)| b.block.number);
        debug!(target: "live-trie::persistence", ?count, ?first, ?last, "Writing batch to storage");

        // Store updates and prune in a single transaction
        let result = self.storage.provider_rw().and_then(|provider| {
            // 1. Store the new block updates (without pruning — pass None)
            let write_start = Instant::now();
            let res = provider.store_trie_updates_batch(updates)?;
            let write_duration = write_start.elapsed();

            // 2. Prune old state using the pruner on the same transaction
            let prune_start = Instant::now();
            let prune_result = self.pruner.prune_with_provider(&provider);
            let prune_duration = prune_start.elapsed();

            match &prune_result {
                Ok(output) => {
                    if *output != Default::default() {
                        info!(
                            target: "live-trie::persistence",
                            ?output,
                            "Pruning complete within save transaction"
                        );
                    }
                }
                Err(e) => {
                    error!(target: "live-trie::persistence", ?e, "Pruning failed during save, aborting transaction");
                }
            }

            // 3. Abort the entire transaction if pruning failed
            prune_result.map_err(|e| crate::OpProofsStorageError::Other(e.to_string()))?;

            // 4. Commit both store and prune atomically
            provider.commit()?;
            Ok((res, write_duration, prune_duration))
        });

        let (successful_last, total_write_count, write_duration, prune_duration) = match result {
            Ok((counts, wd, pd)) => (last, counts, Some(wd), Some(pd)),
            Err(e) => {
                error!(target: "live-trie::persistence", ?e, "Failed to persist batch trie updates");
                (None, WriteCounts::default(), None, None)
            }
        };

        let duration = start.elapsed();
        info!(
            target: "live-trie::persistence",
            ?successful_last,
            ?duration,
            ?write_duration,
            ?prune_duration,
            ?total_write_count,
            blocks_count = count,
            "Batch write complete"
        );
        let _ = reply_tx.send(successful_last);
    }

    fn on_unwind(
        &self,
        to: BlockWithParent,
        reply_tx: Sender<Result<(), ()>>,
    ) {
        debug!(target: "live-trie::persistence", to_block = ?to.block.number, "Unwinding storage");
        let result = self.storage.provider_rw().and_then(|provider| {
            provider.unwind_history(to)?;
            provider.commit()
        });
        match result {
            Ok(_) => {
                debug!(target: "live-trie::persistence", "Unwind successful");
                let _ = reply_tx.send(Ok(()));
            }
            Err(e) => {
                error!(target: "live-trie::persistence", ?e, "Unwind failed");
                let _ = reply_tx.send(Err(()));
            }
        }
    }
}
