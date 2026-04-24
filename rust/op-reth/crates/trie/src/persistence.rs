//! Persistence implementation for external proof

use crate::{
    api::{OpProofsProviderRw, WriteCounts},
    prune::OpProofStoragePruner,
    BlockStateDiff, OpProofsStore, OpProofsStorageError,
};
#[cfg(feature = "metrics")]
use crate::metrics::PersistenceMetrics;
use alloy_eips::eip1898::BlockWithParent;
use reth_provider::BlockHashReader;
use crossbeam_channel::{Receiver, Sender};
use std::{sync::Arc, thread, time::Instant};
use tracing::{debug, error, info};

/// Messages sent to the persistence service.
#[derive(Debug)]
pub enum LiveTriePersistenceAction {
    /// Save a batch of trie updates to storage.
    ///
    /// Contains:
    /// 1. The list of blocks and their diffs (ordered Oldest -> Newest).
    /// 2. A response channel to return the highest block number persisted (for pruning).
    SaveUpdates(Vec<Arc<(BlockWithParent, BlockStateDiff)>>, Sender<Option<u64>>),
    /// Unwind history to the specified block (inclusive).
    /// All history strictly after this block is removed.
    Unwind(BlockWithParent, Sender<Result<(), String>>),
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
    pub fn spawn<H, S>(pruner: OpProofStoragePruner<S, H>, storage: S) -> Self
    where
        S: OpProofsStore + Clone + 'static,
        H: BlockHashReader + Send + Sync + 'static,
    {
        let (tx, rx) = crossbeam_channel::bounded(2);
        let service = LiveTriePersistenceService::new(pruner, storage, rx);

        thread::Builder::new()
            .name("Live Trie Persistence".into())
            .spawn(move || service.run())
            .expect("failed to spawn live trie persistence thread");

        Self::new(tx)
    }

    /// Send a save request.
    ///
    /// Returns an error if the persistence service has stopped.
    pub fn save_updates(
        &self,
        updates: Vec<Arc<(BlockWithParent, BlockStateDiff)>>,
        response_tx: Sender<Option<u64>>,
    ) -> Result<(), OpProofsStorageError> {
        self.sender.send(LiveTriePersistenceAction::SaveUpdates(updates, response_tx))
            .map_err(|_| OpProofsStorageError::Other("Persistence service disconnected".into()))
    }

    /// Send an unwind request.
    ///
    /// Returns an error if the persistence service has stopped.
    pub fn unwind(
        &self,
        to: BlockWithParent,
        response_tx: Sender<Result<(), String>>,
    ) -> Result<(), OpProofsStorageError> {
        self.sender.send(LiveTriePersistenceAction::Unwind(to, response_tx))
            .map_err(|_| OpProofsStorageError::Other("Persistence service disconnected".into()))
    }
}

/// Service that runs in a background thread to persist trie updates.
#[derive(Debug)]
pub struct LiveTriePersistenceService<H, S> {
    /// Pruner that also owns the storage backend and block hash reader.
    pruner: OpProofStoragePruner<S, H>,
    storage: S,
    incoming: Receiver<LiveTriePersistenceAction>,

    #[cfg(feature = "metrics")]
    metrics: PersistenceMetrics,
}

impl<H: BlockHashReader, S: OpProofsStore> LiveTriePersistenceService<H, S> {
    /// Create a new persistence service.
    pub fn new(pruner: OpProofStoragePruner<S, H>, storage: S, incoming: Receiver<LiveTriePersistenceAction>) -> Self {
        Self {
            pruner,
            storage,
            incoming,

            #[cfg(feature = "metrics")]
            metrics: PersistenceMetrics::new_with_labels(&[] as &[(&str, &str)]),
        }
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
        arc_updates: Vec<Arc<(BlockWithParent, BlockStateDiff)>>,
        reply_tx: Sender<Option<u64>>,
    ) {
        if arc_updates.is_empty() {
            let _ = reply_tx.send(None);
            return;
        }

        let start = Instant::now();
        let count = arc_updates.len();
        let first = arc_updates.first().map(|arc| arc.0.block.number);
        let last = arc_updates.last().map(|arc| arc.0.block.number);
        debug!(target: "live-trie::persistence", ?count, ?first, ?last, "Writing batch to storage");

        // Convert from Arc to owned on the persistence thread (not the caller thread)
        // to avoid blocking block execution with deep clones.
        let updates: Vec<(BlockWithParent, BlockStateDiff)> = arc_updates
            .into_iter()
            .map(|arc| Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
            .collect();

        // Store updates and prune in a single transaction
        let provider_rw_start = Instant::now();
        let result = self.storage.provider_rw().and_then(|provider| {
            let open_tx_duration = provider_rw_start.elapsed();

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
            let commit_start = Instant::now();
            provider.commit()?;
            let commit_duration = commit_start.elapsed();

            Ok((res, open_tx_duration, write_duration, prune_duration, commit_duration))
        });

        let (successful_last, total_write_count, open_tx_duration, write_duration, prune_duration, commit_duration) = match result {
            Ok((counts, otd, wd, pd, cd)) => (last, counts, Some(otd), Some(wd), Some(pd), Some(cd)),
            Err(e) => {
                error!(target: "live-trie::persistence", ?e, "Failed to persist batch trie updates");
                (None, WriteCounts::default(), None, None, None, None)
            }
        };

        #[cfg(feature = "metrics")]
        {
            self.metrics.increment_write_counts(&total_write_count);
            if let Some(d) = open_tx_duration {
                self.metrics.open_tx_duration_seconds.record(d);
            }
            if let Some(d) = write_duration {
                self.metrics.write_duration_seconds.record(d);
            }
            if let Some(d) = prune_duration {
                self.metrics.prune_duration_seconds.record(d);
            }
            if let Some(d) = commit_duration {
                self.metrics.commit_duration_seconds.record(d);
            }
        }

        let duration = start.elapsed();
        info!(
            target: "live-trie::persistence",
            ?successful_last,
            ?duration,
            ?open_tx_duration,
            ?write_duration,
            ?prune_duration,
            ?commit_duration,
            ?total_write_count,
            blocks_count = count,
            "Batch write complete"
        );
        let _ = reply_tx.send(successful_last);
    }

    fn on_unwind(
        &self,
        to: BlockWithParent,
        reply_tx: Sender<Result<(), String>>,
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
                let _ = reply_tx.send(Err(e.to_string()));
            }
        }
    }
}
