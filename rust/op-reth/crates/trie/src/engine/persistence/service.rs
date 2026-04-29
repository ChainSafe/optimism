//! Background persistence service for the live trie engine.

use crate::{
    api::{OpProofsProviderRw, WriteCounts},
    prune::OpProofStoragePruner,
    BlockStateDiff, OpProofsStore,
};
#[cfg(feature = "metrics")]
use super::metrics::PersistenceMetrics;
use alloy_eips::eip1898::BlockWithParent;
use crossbeam_channel::{Receiver, Sender};
use reth_provider::BlockHashReader;
use std::{sync::Arc, time::Instant};
use tracing::{debug, error, info};
use super::handle::PersistenceAction;

/// Service that runs in a background thread to persist trie updates.
#[derive(Debug)]
pub(crate) struct PersistenceService<H, S> {
    /// Pruner that also owns the storage backend and block hash reader.
    pruner: OpProofStoragePruner<S, H>,
    storage: S,
    incoming: Receiver<PersistenceAction>,

    #[cfg(feature = "metrics")]
    metrics: PersistenceMetrics,
}

impl<H: BlockHashReader, S: OpProofsStore> PersistenceService<H, S> {
    /// Create a new persistence service.
    pub(crate) fn new(
        pruner: OpProofStoragePruner<S, H>,
        storage: S,
        incoming: Receiver<PersistenceAction>,
    ) -> Self {
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
    pub(crate) fn run(self) {
        debug!(target: "live-trie::persistence", "Service started");

        while let Ok(action) = self.incoming.recv() {
            match action {
                PersistenceAction::Unwind(to, reply_tx) => {
                    self.on_unwind(to, reply_tx);
                }
                PersistenceAction::SaveUpdates(updates, reply_tx) => {
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

        let (successful_last, total_write_count, open_tx_duration, write_duration, prune_duration, commit_duration) =
            match result {
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

    fn on_unwind(&self, to: BlockWithParent, reply_tx: Sender<Result<(), String>>) {
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
