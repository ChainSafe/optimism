//! Error type for the live trie collector engine.

use crate::OpProofsStorageError;
use reth_execution_errors::BlockExecutionError;
use reth_provider::ProviderError;
use thiserror::Error;
use alloy_primitives::B256;

/// Errors produced by the live trie collector engine.
///
/// Distinct from [`OpProofsStorageError`] which covers the storage layer.
/// Engine errors cover orchestration failures: sync, persistence lifecycle,
/// and inter-thread communication.
#[derive(Debug, Error)]
pub enum EngineError {
    /// Block was not found in the provider during sync catch-up.
    #[error("Block {0} not found in provider")]
    BlockNotFound(u64),
    /// The background persistence service channel closed unexpectedly.
    #[error("Persistence service disconnected")]
    PersistenceDisconnected,
    /// A persistence save or unwind operation timed out.
    #[error("Persistence operation timed out")]
    PersistenceTimeout,
    /// The persistence service reported an unwind failure.
    #[error("Unwind failed in persistence service: {0}")]
    UnwindFailed(String),
    /// The collector engine thread terminated unexpectedly.
    #[error("Collector engine terminated unexpectedly")]
    EngineDied,
    /// Block is at the correct number (tip+1) but its parent hash does not match the current tip.
    /// This indicates the block belongs to a different fork.
    #[error(
        "Parent hash mismatch at block {block_number}: expected {expected_parent_hash}, got {actual_parent_hash}"
    )]
    /// The computed state root after EVM execution does not match the block header's state root.
    ParentHashMismatch {
        /// The block number where the mismatch occurred
        block_number: u64,
        /// The expected parent hash (current tip hash)
        expected_parent_hash: B256,
        /// The actual parent hash from the block header
        actual_parent_hash: B256,
    },
    /// The computed state root after EVM execution does not match the block header's state root.
    #[error(
        "State root mismatch for block {block_number} (have: {current_state_hash}, expected: {expected_state_hash})"
    )]
    /// The computed state root after EVM execution does not match the block header's state root.
    StateRootMismatch {
        /// The block number where the mismatch occurred
        block_number: u64,
        /// The actual state root computed from execution
        current_state_hash: B256,
        /// The expected state root from the block header
        expected_state_hash: B256,
    },
    /// A block execution error during EVM execution.
    #[error(transparent)]
    Execution(#[from] BlockExecutionError),
    /// A provider error propagated from the block provider.
    #[error(transparent)]
    Provider(#[from] ProviderError),
    /// A storage-layer error propagated from the underlying store or provider.
    #[error(transparent)]
    Storage(#[from] OpProofsStorageError),
}

impl From<EngineError> for OpProofsStorageError {
    fn from(e: EngineError) -> Self {
        match e {
            EngineError::Storage(inner) => inner,
            EngineError::Provider(inner) => inner.into(),
            EngineError::Execution(inner) => inner.into(),
            other => Self::Other(other.to_string()),
        }
    }
}
