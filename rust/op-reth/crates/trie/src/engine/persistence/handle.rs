//! Handle and action enum for the persistence service.

use crate::{BlockStateDiff, OpProofsStore, prune::OpProofStoragePruner};
use super::super::error::EngineError;
use alloy_eips::eip1898::BlockWithParent;
use crossbeam_channel::Sender;
use reth_provider::BlockHashReader;
use std::{sync::Arc, thread};
use super::service::PersistenceService;

/// Messages sent to the persistence service.
#[derive(Debug)]
pub(crate) enum PersistenceAction {
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
pub(crate) struct PersistenceHandle {
    sender: Sender<PersistenceAction>,
}

impl PersistenceHandle {
    /// Create a new handle.
    pub(crate) fn new(sender: Sender<PersistenceAction>) -> Self {
        Self { sender }
    }

    /// Spawn the service in a new thread and return a handle.
    pub(crate) fn spawn<H, S>(pruner: OpProofStoragePruner<S, H>, storage: S) -> Self
    where
        S: OpProofsStore + Clone + 'static,
        H: BlockHashReader + Send + Sync + 'static,
    {
        let (tx, rx) = crossbeam_channel::bounded(2);
        let service = PersistenceService::new(pruner, storage, rx);

        thread::Builder::new()
            .name("Live Trie Persistence".into())
            .spawn(move || service.run())
            .expect("failed to spawn live trie persistence thread");

        Self::new(tx)
    }

    /// Send a save request.
    ///
    /// Returns an error if the persistence service has stopped.
    pub(crate) fn save_updates(
        &self,
        updates: Vec<Arc<(BlockWithParent, BlockStateDiff)>>,
        response_tx: Sender<Option<u64>>,
    ) -> Result<(), EngineError> {
        self.sender
            .send(PersistenceAction::SaveUpdates(updates, response_tx))
            .map_err(|_| EngineError::PersistenceDisconnected)
    }

    /// Send an unwind request.
    ///
    /// Returns an error if the persistence service has stopped.
    pub(crate) fn unwind(
        &self,
        to: BlockWithParent,
        response_tx: Sender<Result<(), String>>,
    ) -> Result<(), EngineError> {
        self.sender
            .send(PersistenceAction::Unwind(to, response_tx))
            .map_err(|_| EngineError::PersistenceDisconnected)
    }
}
