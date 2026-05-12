//! [`SnapshotInitJob`] — builds a one-time trie-state snapshot for fast backfill.
//!
//! The snapshot mirrors the trie state at a caller-supplied target block into
//! the parallel `V2*TrieSnapshot` tables. Once `Ready`, the snapshot lets the
//! backfill compute phase skip the V2 merge-walk (and its per-key
//! `find_source` work) — see [`crate::backfill`] for the broader rationale.
//!
//! ## Phases (mirrored by methods on [`SnapshotInitJob`])
//!
//! 1. [`prepare_anchor`](SnapshotInitJob::prepare_anchor) — validate the
//!    target falls inside the proofs window and resolve its hash.
//! 2. [`classify_or_plant_meta`](SnapshotInitJob::classify_or_plant_meta) —
//!    decide whether to resume an in-progress build or start fresh; if fresh,
//!    plant `Building` meta in its own short rw-tx so an interrupt before the
//!    first data chunk still leaves a valid resume anchor.
//! 3. [`drain_account_trie`](SnapshotInitJob::drain_account_trie) — chunked
//!    copy of the account trie at the target block into the snapshot table.
//! 4. [`drain_storage_trie`](SnapshotInitJob::drain_storage_trie) — chunked
//!    copy of the storage trie at the target block, walking hashed accounts
//!    and draining each account's storage cursor.
//! 5. [`validate_state_root`](SnapshotInitJob::validate_state_root) — compute
//!    state root from the snapshot tables + live hashed leaves, compare
//!    against the reth header.
//! 6. [`finalize_ready`](SnapshotInitJob::finalize_ready) — flip status to
//!    `Ready` and commit.
//!
//! ## Restart / resume
//!
//! Each chunk commits independently; after a crash the meta stays at
//! [`SnapshotStatus::Building`] with the original anchor. A re-run inspects
//! [`OpProofsSnapshotInitProvider::snapshot_init_anchor`], discovers the
//! resume keys from the partially-populated destination tables, and continues
//! from there. Resume is only safe when the target block matches the
//! existing anchor — otherwise the init aborts with
//! [`BackfillError::SnapshotResumeDriftDetected`].
//!
//! [`SnapshotStatus::Ready`]: crate::db::SnapshotStatus::Ready
//! [`SnapshotStatus::Building`]: crate::db::SnapshotStatus::Building
//! [`BackfillError::SnapshotResumeDriftDetected`]: crate::BackfillError::SnapshotResumeDriftDetected

use crate::{
    BackfillError, OpProofsHashedAccountCursorFactory, OpProofsProviderRO,
    OpProofsSnapshotInitProvider, SnapshotTrieCursorFactory,
    db::{SnapshotMeta, SnapshotStatus},
};
use alloy_eips::BlockNumHash;
use alloy_primitives::{B256, BlockNumber};
use derive_more::Constructor;
use reth_primitives_traits::AlloyBlockHeader;
use reth_provider::{BlockHashReader, HeaderProvider, ProviderError};
use reth_trie::{
    BranchNodeCompact, HashedPostState, Nibbles, StateRoot, StoredNibbles, StoredNibblesSubKey,
    hashed_cursor::{HashedCursor, HashedPostStateCursorFactory},
    trie_cursor::TrieCursor,
};
use std::time::Instant;
use tracing::{debug, info};

/// Rows copied per chunked init transaction.
const SNAPSHOT_INIT_CHUNK_SIZE: usize = 50_000;

/// Output of a successful [`SnapshotInitJob::run`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotInitOutcome {
    /// Snapshot meta written on success.
    pub meta: SnapshotMeta,
    /// Number of account trie nodes copied during this run (does **not**
    /// include rows already present from a prior resumable run).
    pub account_nodes_copied: u64,
    /// Number of storage trie nodes copied during this run.
    pub storage_nodes_copied: u64,
}

/// Builds the one-time trie-state snapshot used by [`crate::backfill::BackfillJob`]
/// to skip the V2 merge-walk.
#[derive(Debug, Constructor)]
pub struct SnapshotInitJob<P, S: crate::OpProofsSnapshotStore + Send> {
    /// Reth DB provider (used to look up the target block's hash + header).
    provider: P,
    /// Op-reth proofs storage that owns the snapshot tables.
    storage: S,
}

impl<P, S> SnapshotInitJob<P, S>
where
    P: HeaderProvider + BlockHashReader + Send,
    S: crate::OpProofsSnapshotStore + Send,
{
    /// Build a snapshot at `target_block`, validating against the reth header.
    ///
    /// `target_block` must fall inside the proofs window's `[earliest, latest]`.
    /// Auto-resumes a partial `Building` snapshot if the existing anchor
    /// matches; refuses to run if a `Ready` or stale snapshot exists at a
    /// different anchor (the caller must drop it first).
    pub fn run(&self, target_block: BlockNumber) -> Result<SnapshotInitOutcome, BackfillError> {
        let start = Instant::now();

        let anchor = self.prepare_anchor(target_block)?;
        let resume = self.classify_or_plant_meta(anchor)?;
        info!(
            target: "reth::op-proofs::snapshot-init",
            target = target_block,
            resume,
            "Starting snapshot init"
        );

        let expected_root = self.expected_state_root(target_block)?;

        let copy_start = Instant::now();
        let account_nodes_copied = self.drain_account_trie(target_block)?;
        let storage_nodes_copied = self.drain_storage_trie(target_block)?;
        let copy_elapsed = copy_start.elapsed();

        let validate_start = Instant::now();
        self.validate_state_root(target_block, expected_root)?;
        let validate_elapsed = validate_start.elapsed();

        let meta = self.finalize_ready(anchor)?;

        info!(
            target: "reth::op-proofs::snapshot-init",
            target = target_block,
            account_nodes_copied,
            storage_nodes_copied,
            copy_elapsed = ?copy_elapsed,
            validate_elapsed = ?validate_elapsed,
            total_elapsed = ?start.elapsed(),
            "Snapshot init complete"
        );

        Ok(SnapshotInitOutcome { meta, account_nodes_copied, storage_nodes_copied })
    }

    /// Validate `target_block` is inside the proofs window and resolve its hash.
    fn prepare_anchor(&self, target_block: BlockNumber) -> Result<BlockNumHash, BackfillError> {
        let ro = self.storage.provider_ro()?;
        let (earliest, _) =
            ro.get_earliest_block_number()?.ok_or(BackfillError::SnapshotInitNoEarliest)?;
        let (latest, _) =
            ro.get_latest_block_number()?.ok_or(BackfillError::SnapshotInitNoEarliest)?;
        if target_block < earliest || target_block > latest {
            return Err(BackfillError::SnapshotInitTargetOutsideWindow {
                target_block,
                earliest,
                latest,
            });
        }

        let target_hash = self
            .provider
            .block_hash(target_block)?
            .ok_or_else(|| ProviderError::HeaderNotFound(target_block.into()))?;
        Ok(BlockNumHash::new(target_block, target_hash))
    }

    /// Classify any existing snapshot meta:
    /// - `None` → start fresh; plant a `Building` row at `anchor` in its own rw-tx.
    /// - `Some(Building, matching anchor)` → resume.
    /// - `Some(Building, different anchor)` → drift; refuse.
    /// - `Some(_)` → already exists; refuse.
    ///
    /// Returns `true` if resuming, `false` if a fresh build.
    fn classify_or_plant_meta(&self, anchor: BlockNumHash) -> Result<bool, BackfillError> {
        let sp = self.storage.snapshot_provider()?;
        let init_anchor = sp.snapshot_init_anchor()?;
        let resume = match init_anchor.meta {
            None => false,
            Some(meta)
                if meta.status == SnapshotStatus::Building && meta.earliest == anchor =>
            {
                true
            }
            Some(meta) if meta.status == SnapshotStatus::Building => {
                return Err(BackfillError::SnapshotResumeDriftDetected {
                    anchor_block: meta.earliest.number,
                    reason:
                        "snapshot init target does not match the in-progress snapshot anchor",
                });
            }
            Some(meta) => {
                return Err(BackfillError::SnapshotAlreadyExists {
                    existing_block: meta.earliest.number,
                    existing_status: meta.status,
                });
            }
        };
        if !resume {
            sp.set_snapshot_init_anchor(anchor)?;
            OpProofsSnapshotInitProvider::commit(sp)?;
        }
        Ok(resume)
    }

    /// Look up the expected state root for `target_block` from reth's headers.
    fn expected_state_root(&self, target_block: BlockNumber) -> Result<B256, BackfillError> {
        Ok(self
            .provider
            .header_by_number(target_block)?
            .ok_or_else(|| ProviderError::HeaderNotFound(target_block.into()))?
            .state_root())
    }

    /// Drain the history-aware account trie cursor at `target_block` into
    /// `V2AccountsTrieSnapshot`, one chunk per rw-tx. Resumes past whatever's
    /// currently in the snapshot.
    ///
    /// Returns the number of rows copied during *this* call (excluding rows
    /// already present from prior runs).
    fn drain_account_trie(&self, target_block: BlockNumber) -> Result<u64, BackfillError> {
        let mut copied = 0u64;
        loop {
            let chunk = {
                let sp = self.storage.snapshot_provider()?;
                let resume_after = sp.snapshot_init_anchor()?.last_account_trie_key;
                let ro = self.storage.provider_ro()?;
                let mut cursor = ro.account_trie_cursor(target_block)?;
                collect_account_chunk(&mut cursor, resume_after, SNAPSHOT_INIT_CHUNK_SIZE)?
            };
            if chunk.is_empty() {
                break;
            }
            let n = chunk.len() as u64;
            let sp = self.storage.snapshot_provider()?;
            sp.store_account_trie_snapshot_branches(chunk)?;
            OpProofsSnapshotInitProvider::commit(sp)?;
            copied += n;
            debug!(
                target: "reth::op-proofs::snapshot-init",
                phase = "accounts",
                chunk = n,
                cumulative = copied,
                "Wrote chunk"
            );
        }
        Ok(copied)
    }

    /// Walk hashed accounts at `target_block`, drain each account's historical
    /// storage trie cursor into `V2StoragesTrieSnapshot`, one chunk per rw-tx.
    ///
    /// Resume tracks the last `(addr, subkey)` written.
    fn drain_storage_trie(&self, target_block: BlockNumber) -> Result<u64, BackfillError> {
        let mut copied = 0u64;
        loop {
            let chunk = {
                let sp = self.storage.snapshot_provider()?;
                let resume_after = sp.snapshot_init_anchor()?.last_storage_trie_key;
                let ro = self.storage.provider_ro()?;
                collect_storage_chunk(&ro, target_block, resume_after, SNAPSHOT_INIT_CHUNK_SIZE)?
            };
            if chunk.is_empty() {
                break;
            }
            let n = chunk.len() as u64;
            let sp = self.storage.snapshot_provider()?;
            sp.store_storage_trie_snapshot_branches(chunk)?;
            OpProofsSnapshotInitProvider::commit(sp)?;
            copied += n;
            debug!(
                target: "reth::op-proofs::snapshot-init",
                phase = "storages",
                chunk = n,
                cumulative = copied,
                "Wrote chunk"
            );
        }
        Ok(copied)
    }

    /// Compute the state root from the snapshot tables and the live hashed
    /// leaves and compare against `expected_root`.
    ///
    /// On mismatch the meta is **not** advanced — it stays at `Building` so
    /// a re-run can diagnose / resume / `snapshot-drop`.
    fn validate_state_root(
        &self,
        target_block: BlockNumber,
        expected_root: B256,
    ) -> Result<(), BackfillError> {
        let sp = self.storage.snapshot_provider()?;
        let state_sorted = HashedPostState::default().into_sorted();
        let computed_root = StateRoot::new(
            SnapshotTrieCursorFactory::new(&sp),
            HashedPostStateCursorFactory::new(
                OpProofsHashedAccountCursorFactory::new(&sp, target_block),
                &state_sorted,
            ),
        )
        .root()?;

        if computed_root != expected_root {
            return Err(BackfillError::StateRootMismatch {
                block_number: target_block,
                computed: computed_root,
                expected: expected_root,
            });
        }
        Ok(())
    }

    /// Flip status to `Ready` and commit in a final rw-tx.
    fn finalize_ready(&self, anchor: BlockNumHash) -> Result<SnapshotMeta, BackfillError> {
        let sp = self.storage.snapshot_provider()?;
        sp.commit_snapshot()?;
        OpProofsSnapshotInitProvider::commit(sp)?;
        Ok(SnapshotMeta::new(anchor, SnapshotStatus::Ready))
    }
}

/// Drain up to `max_entries` rows from a `TrieCursor` strictly after `resume_after`.
fn collect_account_chunk<C: TrieCursor>(
    cursor: &mut C,
    resume_after: Option<StoredNibbles>,
    max_entries: usize,
) -> Result<Vec<(StoredNibbles, BranchNodeCompact)>, BackfillError> {
    if max_entries == 0 {
        return Ok(Vec::new());
    }
    let mut next = match resume_after {
        None => cursor.seek(Nibbles::default())?,
        Some(after) => {
            // `seek` returns the first key >= `after`. If it matches exactly,
            // skip past it; otherwise we're already past.
            match cursor.seek(after.0)? {
                Some((k, _)) if k == after.0 => cursor.next()?,
                other => other,
            }
        }
    };
    let mut out = Vec::with_capacity(max_entries);
    while let Some((k, v)) = next {
        if out.len() >= max_entries {
            break;
        }
        out.push((StoredNibbles(k), v));
        next = cursor.next()?;
    }
    Ok(out)
}

/// Collect a chunk of storage-trie entries by walking hashed accounts at
/// `target_block` and draining each account's storage trie cursor.
///
/// Resume semantics: `resume_after` is the last `(addr, subkey)` already
/// written to the snapshot. We seek the account cursor to `addr`, position
/// that account's storage cursor past `subkey`, drain to chunk-limit, then
/// advance to the next account and so on.
fn collect_storage_chunk<P>(
    proofs_ro: &P,
    target_block: BlockNumber,
    resume_after: Option<(B256, StoredNibblesSubKey)>,
    max_entries: usize,
) -> Result<Vec<(B256, StoredNibblesSubKey, BranchNodeCompact)>, BackfillError>
where
    P: OpProofsProviderRO,
{
    if max_entries == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(max_entries);

    let (start_addr, mut subkey_resume) = match resume_after {
        None => (B256::ZERO, None),
        Some((addr, sk)) => (addr, Some(sk)),
    };

    let mut acc_cursor = proofs_ro.account_hashed_cursor(target_block)?;
    let mut next_account = acc_cursor.seek(start_addr)?;

    while let Some((addr, _account)) = next_account {
        let mut stor_cursor = proofs_ro.storage_trie_cursor(addr, target_block)?;

        // Position past any pending subkey resume (only applies to the first
        // account on this call — subsequent accounts always start at the
        // beginning of their trie).
        let mut next_stor = match subkey_resume.take() {
            None => stor_cursor.seek(Nibbles::default())?,
            Some(s) => {
                match stor_cursor.seek(s.0)? {
                    Some((k, _)) if k == s.0 => stor_cursor.next()?,
                    other => other,
                }
            }
        };

        while let Some((subkey, node)) = next_stor {
            if out.len() >= max_entries {
                return Ok(out);
            }
            out.push((addr, StoredNibblesSubKey(subkey), node));
            next_stor = stor_cursor.next()?;
        }

        next_account = acc_cursor.next()?;
    }

    Ok(out)
}
