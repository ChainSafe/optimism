//! Error type for backfill operations.

use crate::OpProofsStorageError;
use alloy_primitives::B256;
use reth_execution_errors::StateRootError;
use reth_provider::ProviderError;

/// Error type for backfill operations.
#[derive(Debug, thiserror::Error)]
pub enum BackfillError {
    /// Error bubbled up from proofs storage operations.
    #[error(transparent)]
    Storage(#[from] OpProofsStorageError),
    /// Error from reth provider operations.
    #[error(transparent)]
    Provider(#[from] ProviderError),
    /// State root computation failed.
    #[error(transparent)]
    StateRoot(#[from] StateRootError),
    /// Computed state root does not match the expected root from the header.
    #[error(
        "State root mismatch at block {block_number}: computed {computed:?}, expected {expected:?}"
    )]
    StateRootMismatch {
        /// Block number being validated (the block whose before-state is being checked).
        block_number: u64,
        /// Computed root from the proofs storage overlay.
        computed: B256,
        /// Expected root from reth's block header.
        expected: B256,
    },
    /// The snapshot init job ran but reads from the proofs window returned no
    /// `latest` block — there's nothing to anchor the snapshot to.
    #[error("snapshot init: proofs window has no latest block")]
    SnapshotInitNoLatest,
    /// The snapshot init job found an existing snapshot. The caller must drop
    /// it explicitly before rebuilding to avoid silently discarding state.
    #[error(
        "snapshot init: a snapshot already exists at block {existing_block} with status {existing_status:?}; call clear_snapshot first if you intend to rebuild"
    )]
    SnapshotAlreadyExists {
        /// `meta.earliest.number` of the existing snapshot.
        existing_block: u64,
        /// `meta.status` of the existing snapshot.
        existing_status: crate::db::SnapshotStatus,
    },
    /// A `Building` snapshot exists but cannot be safely resumed.
    ///
    /// Either the proofs window has advanced past the snapshot's anchor
    /// (source state has changed since init started) or the source trie tables
    /// are unexpectedly empty under a Building meta. The operator must
    /// `snapshot-drop` and re-run.
    #[error(
        "snapshot init: cannot resume in-progress snapshot at block {anchor_block} — {reason}"
    )]
    SnapshotResumeDriftDetected {
        /// `meta.earliest.number` of the in-progress snapshot.
        anchor_block: u64,
        /// Human-readable reason the resume was refused.
        reason: &'static str,
    },
    /// `run_with_snapshot` was called but the snapshot is missing or not
    /// aligned with the proofs-window `earliest`. The caller must build /
    /// refresh the snapshot first.
    #[error(
        "snapshot not aligned with proofs window: expected Ready snapshot at block {expected_earliest}, got status {actual_status:?} at block {actual_earliest:?}"
    )]
    SnapshotNotAligned {
        /// The proofs-window `earliest.number` at the time of dispatch.
        expected_earliest: u64,
        /// Snapshot meta's `status`, or `None` if no snapshot exists.
        actual_status: Option<crate::db::SnapshotStatus>,
        /// Snapshot meta's `earliest.number`, or `None` if no snapshot exists.
        actual_earliest: Option<u64>,
    },
}
