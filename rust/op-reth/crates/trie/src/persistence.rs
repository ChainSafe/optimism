//! Persistence implementation for external proof

use crate::{BlockStateDiff, OpProofsStorage, OpProofsStore, api::WriteCounts};
use alloy_eips::{NumHash, eip1898::BlockWithParent};
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
    pub fn spawn<H, S>(block_hash_reader: H, min_block_interval: u64, storage: OpProofsStorage<S>) -> Self
    where
        S: OpProofsStore + Send + Sync + 'static,
        H: BlockHashReader + Send + Sync + 'static,
    {
        let (tx, rx) = crossbeam_channel::unbounded();
        let service = LiveTriePersistenceService::new(block_hash_reader, min_block_interval, storage, rx);

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
    /// Reader to fetch block hash by block number
    block_hash_reader: H,
    /// Keep at least these many recent blocks
    min_block_interval: u64,
    // The storage backend for proofs data.
    storage: OpProofsStorage<S>,
    incoming: Receiver<LiveTriePersistenceAction>,
}

impl<H: BlockHashReader, S: OpProofsStore> LiveTriePersistenceService<H, S> {
    /// Create a new persistence service.
    pub fn new(block_hash_reader: H, min_block_interval: u64,  storage: OpProofsStorage<S>, incoming: Receiver<LiveTriePersistenceAction>) -> Self {
        Self { block_hash_reader, min_block_interval, storage, incoming }
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

        // Use the batch storage function for atomicity and performance
        let target = if let Some((last_block, _)) = updates.last() {
            let last_block_num = last_block.block.number;
            let target_num = last_block_num.saturating_sub(self.min_block_interval);

            // Check if we actually need to prune
            // If the earliest block is already >= target_num, we don't need to do anything
            let earliest_block = self.storage.get_earliest_block_number().unwrap_or(None);
            if let Some((earliest_num, _)) = earliest_block {
                if earliest_num >= target_num {
                    debug!(target: "live-trie::persistence", earliest_num, target_num, "Earliest block already ahead of target, skipping pruning calculation");
                    // Return None to skip pruning
                    // MdbxProofsStorage::store_trie_updates_batch handles None by doing nothing for pruning
                    None
                } else {
                     let target_hash = self.block_hash_reader.block_hash(target_num).ok().flatten();
                     let parent_hash = self.block_hash_reader.block_hash(target_num.saturating_sub(1)).ok().flatten();

                     if let (Some(hash), Some(parent)) = (target_hash, parent_hash) {
                         Some(BlockWithParent {
                             block: NumHash { number: target_num, hash },
                             parent,
                         })
                     } else {
                         None
                     }
                }
            } else {
                 None
            }
        } else {
             None
        };

        let (successful_last, total_write_count) = match self.storage.store_trie_updates_batch(updates, target) {
            Ok(counts) => (last, counts),
            Err(e) => {
                error!(target: "live-trie::persistence", ?e, "Failed to persist batch trie updates");
                // If the batch fails, we assume nothing was persisted locally (implied by transaction rollback)
                (None, WriteCounts::default())
            }
        };

        let duration = start.elapsed();
        info!(
            target: "live-trie::persistence",
            ?successful_last,
            ?duration,
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
        match self.storage.unwind_history(to) {
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
