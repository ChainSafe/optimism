//! [`SnapshotInitJob`] — builds a one-time trie-state snapshot for fast backfill.
//!
//! The snapshot mirrors `V2AccountsTrie` / `V2StoragesTrie` at the proofs
//! window's current `latest` block, into the parallel `V2*TrieSnapshot`
//! tables. Once `Ready`, the snapshot lets the backfill compute phase skip
//! the V2 merge-walk (and its per-key `find_source` work) — see
//! [`crate::backfill`] for the broader rationale.
//!
//! ## What it does
//!
//! 1. Reads the proofs window's current `latest` block.
//! 2. Copies the source trie tables into the snapshot tables in **chunked
//!    transactions** (one rw-tx per chunk), so the writer never holds long
//!    and the work survives interruption.
//! 3. Computes the state root from the snapshot tables and the live
//!    hashed-leaf tables (the latter are already authoritative at `latest`).
//! 4. Compares against the header's `state_root` for `latest`. On mismatch
//!    the meta stays at `Building` so a retry can diagnose / re-run.
//! 5. On success, marks the meta row [`SnapshotStatus::Ready`] and commits.
//!
//! ## Restart / resume
//!
//! Each chunk commits independently; after a crash the meta stays at
//! [`SnapshotStatus::Building`] with the original anchor. A re-run inspects
//! [`OpProofsSnapshotInitProvider::snapshot_init_anchor`], discovers the
//! resume keys from the partially-populated destination tables, and continues
//! from there.
//!
//! Resume is only safe when the proofs-window `latest` still equals the
//! anchor — if the node has processed new blocks while init was paused, the
//! source has drifted and the partial snapshot is no longer consistent.
//! In that case the init aborts with [`BackfillError::SnapshotResumeDriftDetected`]
//! and the operator must drop the partial snapshot to rebuild.
//!
//! [`SnapshotStatus::Ready`]: crate::db::SnapshotStatus::Ready
//! [`SnapshotStatus::Building`]: crate::db::SnapshotStatus::Building
//! [`BackfillError::SnapshotResumeDriftDetected`]: crate::BackfillError::SnapshotResumeDriftDetected

use crate::{
    BackfillError, OpProofsHashedAccountCursorFactory, OpProofsProviderRO,
    OpProofsSnapshotInitProvider, OpProofsStore, SnapshotTrieCursorFactory,
    db::{SnapshotMeta, SnapshotStatus},
};
use alloy_eips::BlockNumHash;
use derive_more::Constructor;
use reth_primitives_traits::AlloyBlockHeader;
use reth_provider::{HeaderProvider, ProviderError};
use reth_trie::{HashedPostState, StateRoot, hashed_cursor::HashedPostStateCursorFactory};
use std::time::Instant;
use tracing::{debug, info};

/// Rows copied per chunked init transaction. Sized so each tx writes on the
/// order of a few MB (entries are ~100 bytes each), keeping individual commit
/// costs low while still amortising rw-tx open/close overhead.
const SNAPSHOT_INIT_CHUNK_SIZE: usize = 50_000;

/// Output of a successful [`SnapshotInitJob::run`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotInitOutcome {
    /// Snapshot meta written on success.
    pub meta: SnapshotMeta,
    /// Number of account trie nodes copied during this run (does **not**
    /// include rows that were already present from a prior resumable run).
    pub account_nodes_copied: u64,
    /// Number of storage trie nodes copied during this run.
    pub storage_nodes_copied: u64,
}

/// Build a [`SnapshotStatus::Ready`] snapshot at the proofs-window's current
/// `latest` block.
///
/// This is the core logic of [`SnapshotInitJob::run`], factored out so the
/// [`crate::backfill::BackfillJob`] auto-strategy can invoke it without
/// moving its own `provider` and `storage` fields. Auto-resumes a partial
/// `Building` snapshot if the anchor still matches `latest`; otherwise
/// refuses to run.
pub(crate) fn build_snapshot_at_latest<P, S>(
    reth_provider: &P,
    storage: &S,
) -> Result<SnapshotInitOutcome, BackfillError>
where
    P: HeaderProvider + Send,
    S: OpProofsStore + Send,
    // `OpProofsSnapshotReader` is needed for the final state-root validation
    // (the snapshot cursor factory reads from the same provider). `OpProofsProviderRO`
    // is implied via `OpProofsStore::BackfillProvider`'s super-bound but spelt
    // out here so the hashed-account cursor factory compiles.
    for<'a> S::BackfillProvider<'a>: OpProofsSnapshotInitProvider + crate::OpProofsSnapshotReader,
{
    let start = Instant::now();

    let (latest_number, latest_hash) = {
        let ro = storage.provider_ro()?;
        ro.get_latest_block_number()?.ok_or(BackfillError::SnapshotInitNoLatest)?
    };
    let anchor = BlockNumHash::new(latest_number, latest_hash);

    // Classify existing snapshot state via a single anchor read.
    let resume = {
        let bp = storage.backfill_provider()?;
        let init_anchor = bp.snapshot_init_anchor()?;
        match init_anchor.meta {
            None => false,
            Some(meta) if meta.status == SnapshotStatus::Building &&
                meta.earliest == anchor =>
            {
                // Anchor matches `latest` — source hasn't drifted, safe to resume.
                // The final state-root validation catches any subtler corruption.
                true
            }
            Some(meta) if meta.status == SnapshotStatus::Building => {
                return Err(BackfillError::SnapshotResumeDriftDetected {
                    anchor_block: meta.earliest.number,
                    reason:
                        "proofs-window `latest` has moved past the in-progress snapshot anchor",
                });
            }
            Some(meta) => {
                return Err(BackfillError::SnapshotAlreadyExists {
                    existing_block: meta.earliest.number,
                    existing_status: meta.status,
                });
            }
        }
    };

    info!(
        target: "reth::op-proofs::snapshot-init",
        latest = latest_number,
        resume,
        "Starting snapshot init"
    );

    let expected_root = reth_provider
        .header_by_number(latest_number)?
        .ok_or_else(|| ProviderError::HeaderNotFound(latest_number.into()))?
        .state_root();

    // 1. On a fresh start, plant `Building` meta in its own short rw-tx so an
    //    interrupt before the first data chunk still leaves a valid resume
    //    anchor.
    if !resume {
        let bp = storage.backfill_provider()?;
        bp.set_snapshot_meta(SnapshotMeta::new(anchor, SnapshotStatus::Building))?;
        OpProofsSnapshotInitProvider::commit(bp)?;
    }

    // 2. Account-trie phase: drain V2AccountsTrie → V2AccountsTrieSnapshot one
    //    chunk per rw-tx, picking up after whatever's currently in the snapshot.
    let mut account_nodes_copied = 0u64;
    let copy_start = Instant::now();
    loop {
        let resume_after = storage.backfill_provider()?.snapshot_init_anchor()?.last_account_trie_key;
        let chunk = storage
            .backfill_provider()?
            .account_trie_source_chunk(resume_after, SNAPSHOT_INIT_CHUNK_SIZE)?;
        if chunk.is_empty() {
            break;
        }
        let n = chunk.len() as u64;
        let bp = storage.backfill_provider()?;
        bp.store_account_trie_snapshot_branches(chunk)?;
        OpProofsSnapshotInitProvider::commit(bp)?;
        account_nodes_copied += n;
        debug!(
            target: "reth::op-proofs::snapshot-init",
            phase = "accounts",
            chunk = n,
            cumulative = account_nodes_copied,
            "Wrote chunk"
        );
    }

    // 3. Storage-trie phase: same pattern over V2StoragesTrie.
    let mut storage_nodes_copied = 0u64;
    loop {
        let resume_after = storage.backfill_provider()?.snapshot_init_anchor()?.last_storage_trie_key;
        let chunk = storage
            .backfill_provider()?
            .storage_trie_source_chunk(resume_after, SNAPSHOT_INIT_CHUNK_SIZE)?;
        if chunk.is_empty() {
            break;
        }
        let n = chunk.len() as u64;
        let bp = storage.backfill_provider()?;
        bp.store_storage_trie_snapshot_branches(chunk)?;
        OpProofsSnapshotInitProvider::commit(bp)?;
        storage_nodes_copied += n;
        debug!(
            target: "reth::op-proofs::snapshot-init",
            phase = "storages",
            chunk = n,
            cumulative = storage_nodes_copied,
            "Wrote chunk"
        );
    }
    let copy_elapsed = copy_start.elapsed();

    // 4. Validate state root + flip to Ready in one final rw-tx.
    let bp = storage.backfill_provider()?;
    let validate_start = Instant::now();
    let state_sorted = HashedPostState::default().into_sorted();
    let computed_root = StateRoot::new(
        SnapshotTrieCursorFactory::new(&bp),
        HashedPostStateCursorFactory::new(
            OpProofsHashedAccountCursorFactory::new(&bp, latest_number),
            &state_sorted,
        ),
    )
    .root()?;
    let validate_elapsed = validate_start.elapsed();

    if computed_root != expected_root {
        // Leave the snapshot in Building so a follow-up run can diagnose and
        // either resume or `snapshot-drop`.
        return Err(BackfillError::StateRootMismatch {
            block_number: latest_number,
            computed: computed_root,
            expected: expected_root,
        });
    }

    let meta = SnapshotMeta::new(anchor, SnapshotStatus::Ready);
    bp.set_snapshot_meta(meta)?;
    OpProofsSnapshotInitProvider::commit(bp)?;

    info!(
        target: "reth::op-proofs::snapshot-init",
        latest = latest_number,
        account_nodes_copied,
        storage_nodes_copied,
        copy_elapsed = ?copy_elapsed,
        validate_elapsed = ?validate_elapsed,
        total_elapsed = ?start.elapsed(),
        "Snapshot init complete"
    );

    Ok(SnapshotInitOutcome { meta, account_nodes_copied, storage_nodes_copied })
}

/// Builds the one-time trie-state snapshot used by [`crate::backfill::BackfillJob`]
/// to skip the V2 merge-walk.
#[derive(Debug, Constructor)]
pub struct SnapshotInitJob<P, S: OpProofsStore + Send> {
    /// Reth DB provider (used to look up the expected state root at `latest`).
    provider: P,
    /// Op-reth proofs storage that owns the snapshot tables.
    storage: S,
}

impl<P, S> SnapshotInitJob<P, S>
where
    P: HeaderProvider + Send,
    S: OpProofsStore + Send,
    for<'a> S::BackfillProvider<'a>: OpProofsSnapshotInitProvider + crate::OpProofsSnapshotReader,
{
    /// Build a snapshot at the current `latest` block, validating against the header.
    ///
    /// Auto-resumes a partial `Building` snapshot if the anchor matches;
    /// refuses to run if a `Ready` or stale snapshot exists (the caller must
    /// drop it first).
    pub fn run(&self) -> Result<SnapshotInitOutcome, BackfillError> {
        build_snapshot_at_latest(&self.provider, &self.storage)
    }
}
