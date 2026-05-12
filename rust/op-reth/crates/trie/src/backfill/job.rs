//! [`BackfillJob`] implementation.

use super::{
    changesets::{
        ComputeTimings, compute_block_backfill_diff, compute_block_backfill_diff_with_snapshot,
    },
    error::BackfillError,
};
use crate::{
    OpProofsBackfillProvider, OpProofsProviderRO, OpProofsSnapshotInitProvider,
    OpProofsSnapshotProviderRW, OpProofsSnapshotProviderRO, OpProofsStorageError, OpProofsStore,
    db::SnapshotStatus,
    proof::DatabaseStateRoot,
};
use alloy_eips::{BlockNumHash, eip1898::BlockWithParent};
use alloy_primitives::BlockNumber;
use derive_more::Constructor;
use reth_primitives_traits::AlloyBlockHeader;
use reth_provider::{
    BlockHashReader, BlockNumReader, ChangeSetReader, DBProvider, HeaderProvider, ProviderError,
    StageCheckpointReader, StorageChangeSetReader, StorageSettingsCache,
};
use reth_trie::StateRoot;
use reth_trie_common::HashedPostState;
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// How often to emit a progress line during a long backfill, measured in
/// blocks committed.
const LOG_EVERY: u64 = 1_000;

/// Cumulative time spent in each phase of [`BackfillJob::backfill_block`].
/// Reported alongside the progress line so operators can see which phase
/// dominates a slow backfill. The compute phase is split into its two
/// sub-steps (per-block leaf revert from reth, then proofs-storage trie
/// walk) because they have very different cost characteristics.
#[derive(Debug, Default, Clone, Copy)]
struct PhaseTimings {
    /// `from_reverts_auto(N..=N)` against reth.
    reverts: Duration,
    /// `overlay_root_from_nodes_with_updates` against op-reth's proofs storage.
    trie_changesets: Duration,
    /// `MdbxProofsProviderV2::prepend_block` storage write.
    prepend: Duration,
    /// In-job `StateRoot::overlay_root` validation at block_number - 1.
    validate: Duration,
}

impl PhaseTimings {
    fn add(&mut self, other: Self) {
        self.reverts += other.reverts;
        self.trie_changesets += other.trie_changesets;
        self.prepend += other.prepend;
        self.validate += other.validate;
    }

    /// Per-block average. `done` must be > 0.
    fn averages(&self, done: u64) -> Self {
        let n = done as u32;
        Self {
            reverts: self.reverts / n,
            trie_changesets: self.trie_changesets / n,
            prepend: self.prepend / n,
            validate: self.validate / n,
        }
    }
}

/// Backfill job for proofs storage.
#[derive(Debug, Constructor)]
pub struct BackfillJob<P, S: OpProofsStore + Send> {
    provider: P,
    storage: S,
}

impl<P, S> BackfillJob<P, S>
where
    P: DBProvider
        + StageCheckpointReader
        + ChangeSetReader
        + StorageChangeSetReader
        + BlockNumReader
        + BlockHashReader
        + HeaderProvider
        + StorageSettingsCache
        + Send,
    S: OpProofsStore + Send,
{
    /// Backfill proofs data down to `target_earliest_block`.
    ///
    /// Extends the stored proof window from `[earliest, latest]` backward to
    /// `[target_earliest_block, latest]`. Each block is committed atomically so
    /// the job is restart-safe: on crash, resume from the current `earliest`.
    ///
    /// Returns immediately if `target_earliest_block >= current earliest`.
    pub fn run(&self, target_earliest_block: u64) -> Result<(), BackfillError> {
        let ro = self.storage.provider_ro()?;
        let Some((current_earliest, _)) = ro.get_earliest_block_number()? else {
            return Err(BackfillError::Storage(OpProofsStorageError::NoBlocksFound));
        };
        drop(ro);

        if target_earliest_block >= current_earliest {
            return Ok(());
        }

        let total = current_earliest - target_earliest_block;
        let start = Instant::now();
        let mut phase_totals = PhaseTimings::default();
        // Reset the find_source counters so the first progress window reflects
        // only this backfill run (not any prior reader activity on this thread).
        let _ = crate::db::find_source_stats::snapshot_and_reset();
        info!(
            target: "reth::op-proofs::backfill",
            from = current_earliest,
            to = target_earliest_block,
            total,
            "Starting proofs backfill"
        );

        for block_number in (target_earliest_block + 1..=current_earliest).rev() {
            phase_totals.add(self.backfill_block(block_number)?);

            let done = current_earliest - block_number + 1;
            let is_final = block_number == target_earliest_block + 1;
            if done.is_multiple_of(LOG_EVERY) || is_final {
                let elapsed_secs = start.elapsed().as_secs_f64();
                let blocks_per_sec =
                    if elapsed_secs.is_normal() { done as f64 / elapsed_secs } else { 0.0 };
                let eta_secs = if blocks_per_sec.is_normal() && blocks_per_sec > 0.0 {
                    (total - done) as f64 / blocks_per_sec
                } else {
                    0.0
                };
                let progress_pct = (done as f64 / total as f64) * 100.0;
                let avg = phase_totals.averages(done);
                // Window-local find_source stats: counters reset every progress
                // log, so this reports the FromCurrentState ratio over the last
                // LOG_EVERY blocks — i.e. the "wasted MDBX seek" fraction that
                // a bloom-filter fast-path could eliminate.
                let (from_changeset, from_current_state) =
                    crate::db::find_source_stats::snapshot_and_reset();
                let find_source_total = from_changeset + from_current_state;
                let from_current_state_pct = if find_source_total > 0 {
                    from_current_state as f64 / find_source_total as f64 * 100.0
                } else {
                    0.0
                };
                info!(
                    target: "reth::op-proofs::backfill",
                    done,
                    total,
                    avg_reverts = ?avg.reverts,
                    avg_trie_changesets = ?avg.trie_changesets,
                    avg_prepend = ?avg.prepend,
                    avg_validate = ?avg.validate,
                    find_source_total,
                    from_current_state_pct = format_args!("{from_current_state_pct:.2}"),
                    "progress: {progress_pct:.2}% ({blocks_per_sec:.1} blk/s, ETA {eta_secs:.0}s)"
                );
            }
        }

        let final_avg = phase_totals.averages(total);
        info!(
            target: "reth::op-proofs::backfill",
            blocks = total,
            elapsed = ?start.elapsed(),
            avg_reverts = ?final_avg.reverts,
            avg_trie_changesets = ?final_avg.trie_changesets,
            avg_prepend = ?final_avg.prepend,
            avg_validate = ?final_avg.validate,
            "Proofs backfill complete"
        );

        Ok(())
    }

    /// Backfill a single block `E`: write its historical records and advance `earliest` to `E-1`.
    ///
    /// Returns the wall-clock time spent in each phase, accumulated by
    /// [`Self::run`] into the running averages it reports.
    fn backfill_block(&self, block_number: BlockNumber) -> Result<PhaseTimings, BackfillError> {
        debug!(target: "reth::op-proofs::backfill", block_number, "backfilling block");

        let block_hash = self
            .provider
            .block_hash(block_number)?
            .ok_or_else(|| ProviderError::HeaderNotFound(block_number.into()))?;
        let parent_hash = self
            .provider
            .block_hash(block_number - 1)?
            .ok_or_else(|| ProviderError::HeaderNotFound((block_number - 1).into()))?;

        // Open a fresh RO proofs provider for this iteration: it sees writes
        // committed by the previous `prepend_block`, so its cursor at max=N
        // already reflects state@N. Dropped before opening the RW backfill
        // provider below to avoid holding two transactions on the same env.
        let diff;
        let compute_t: ComputeTimings;
        {
            let proofs_ro = self.storage.provider_ro()?;
            (diff, compute_t) =
                compute_block_backfill_diff(&self.provider, &proofs_ro, block_number)?;
            debug!(
                target: "reth::op-proofs::backfill",
                block_number,
                reverts_elapsed = ?compute_t.reverts,
                trie_changesets_elapsed = ?compute_t.trie_changesets,
                accounts = diff.sorted_post_state.accounts.len(),
                storages = diff.sorted_post_state.storages.len(),
                account_trie_nodes = diff.sorted_trie_updates.account_nodes_ref().len(),
                storage_tries = diff.sorted_trie_updates.storage_tries_ref().len(),
                "computed backfill diff"
            );
        }

        let block_ref = BlockWithParent {
            block: BlockNumHash::new(block_number, block_hash),
            parent: parent_hash,
        };

        let prepend_start = Instant::now();
        let bp = self.storage.backfill_provider()?;
        let counts = bp.prepend_block(block_ref, diff)?;
        let prepend = prepend_start.elapsed();
        debug!(
            target: "reth::op-proofs::backfill",
            block_number,
            elapsed = ?prepend,
            ?counts,
            "prepend_block done"
        );

        // Validate the written before-values by computing a full state root at block_number - 1
        // using the backfill provider (which now includes the prepended data in its transaction).
        // `&bp` implements `OpProofsProviderRO`, so it reads its own uncommitted writes.
        let validate_start = Instant::now();
        let expected_root = self
            .provider
            .header_by_number(block_number - 1)?
            .ok_or_else(|| ProviderError::HeaderNotFound((block_number - 1).into()))?
            .state_root();
        let computed_root =
            StateRoot::overlay_root(&bp, block_number - 1, HashedPostState::default())?;
        let validate = validate_start.elapsed();
        debug!(
            target: "reth::op-proofs::backfill",
            block_number,
            elapsed = ?validate,
            "state root validation done"
        );
        if computed_root != expected_root {
            return Err(BackfillError::StateRootMismatch {
                block_number,
                computed: computed_root,
                expected: expected_root,
            });
        }

        OpProofsBackfillProvider::commit(bp)?;

        Ok(PhaseTimings {
            reverts: compute_t.reverts,
            trie_changesets: compute_t.trie_changesets,
            prepend,
            validate,
        })
    }
}

// ===================== Snapshot-aware backfill path =====================
//
// Available only when the storage's read and backfill providers support the
// snapshot traits. V1 storage does not, and the existing `run` /
// `backfill_block` methods above remain the only path for that backend.

impl<P, S> BackfillJob<P, S>
where
    P: DBProvider
        + StageCheckpointReader
        + ChangeSetReader
        + StorageChangeSetReader
        + BlockNumReader
        + BlockHashReader
        + HeaderProvider
        + StorageSettingsCache
        + Send
        + Sync,
    S: crate::OpProofsSnapshotStore + Send,
    for<'a> S::ProviderRO<'a>: OpProofsSnapshotProviderRO,
    for<'a> S::BackfillProvider<'a>: OpProofsSnapshotProviderRW,
{
    /// Backfill with auto-managed snapshot lifecycle. The recommended entry
    /// point for snapshot-capable storage.
    ///
    /// Behavior:
    /// - If a [`SnapshotStatus::Ready`] snapshot aligned with the current
    ///   proofs-window `earliest` exists, [`Self::run_with_snapshot`] is
    ///   called directly.
    /// - If no snapshot exists, or one exists but is stale / partial / pointing
    ///   at a different anchor, any existing snapshot is cleared and a fresh
    ///   one is built at the current `earliest` via
    ///   [`crate::snapshot::build_snapshot_at_earliest`]. The init reuses the
    ///   merge-walk cursors so it works regardless of where the proofs window
    ///   is anchored.
    pub fn run_auto(&self, target_earliest_block: u64) -> Result<(), BackfillError> {
        let (current_earliest, snapshot_meta) = {
            let ro = self.storage.provider_ro()?;
            let earliest = ro
                .get_earliest_block_number()?
                .ok_or(BackfillError::Storage(OpProofsStorageError::NoBlocksFound))?;
            (earliest.0, ro.snapshot_meta()?)
        };

        if target_earliest_block >= current_earliest {
            return Ok(());
        }

        let aligned = matches!(
            snapshot_meta,
            Some(m) if m.status == SnapshotStatus::Ready
                && m.earliest.number == current_earliest
        );

        if !aligned {
            // Wipe any stale / misaligned / partial snapshot so the rebuild
            // path is clean. The new init walks the merge cursors at
            // `current_earliest`, so it works regardless of where the
            // proofs window is anchored.
            if let Some(existing) = snapshot_meta {
                info!(
                    target: "reth::op-proofs::backfill",
                    existing_status = ?existing.status,
                    existing_earliest = existing.earliest.number,
                    "Snapshot misaligned/partial — dropping before rebuild"
                );
                let sp = self.storage.snapshot_provider()?;
                sp.clear_snapshot()?;
                OpProofsSnapshotInitProvider::commit(sp)?;
            }

            info!(
                target: "reth::op-proofs::backfill",
                anchor = current_earliest,
                "Auto-building snapshot before backfill"
            );
            crate::SnapshotInitJob::new(&self.provider, &self.storage).run(current_earliest)?;
        }

        self.run_with_snapshot(target_earliest_block)
    }

    /// Backfill using the trie snapshot for the per-block compute phase.
    ///
    /// Requires a [`SnapshotStatus::Ready`] snapshot whose
    /// `meta.earliest.number` equals the current proofs-window `earliest` —
    /// callers that want auto-managed lifecycle should use [`Self::run_auto`]
    /// instead.
    ///
    /// Each per-block step:
    /// 1. Computes the trie changeset by reading the snapshot directly (no
    ///    history merge walk).
    /// 2. In one rw-tx: writes the changeset via `prepend_block`, applies the
    ///    revert to the snapshot, and advances `meta.earliest` to `(N-1, parent_hash)`.
    /// 3. Validates the state root at `N-1` against the reth header.
    pub fn run_with_snapshot(&self, target_earliest_block: u64) -> Result<(), BackfillError> {
        // Read both the proofs-window earliest and the snapshot meta in one tx
        // so they reflect a consistent point in time.
        let (current_earliest, snapshot_meta) = {
            let ro = self.storage.provider_ro()?;
            let earliest = ro
                .get_earliest_block_number()?
                .ok_or(BackfillError::Storage(OpProofsStorageError::NoBlocksFound))?;
            let meta = ro.snapshot_meta()?;
            (earliest.0, meta)
        };

        if target_earliest_block >= current_earliest {
            return Ok(());
        }

        // Refuse to run if the snapshot is not aligned with the proofs window.
        // Auto-build / auto-realign lives in Chunk 6's strategy selection.
        match snapshot_meta {
            Some(meta)
                if meta.status == SnapshotStatus::Ready &&
                    meta.earliest.number == current_earliest => {}
            other => {
                return Err(BackfillError::SnapshotNotAligned {
                    expected_earliest: current_earliest,
                    actual_status: other.map(|m| m.status),
                    actual_earliest: other.map(|m| m.earliest.number),
                });
            }
        }

        let total = current_earliest - target_earliest_block;
        let start = Instant::now();
        let mut phase_totals = PhaseTimings::default();
        let _ = crate::db::find_source_stats::snapshot_and_reset();
        info!(
            target: "reth::op-proofs::backfill",
            from = current_earliest,
            to = target_earliest_block,
            total,
            "Starting proofs backfill (snapshot strategy)"
        );

        for block_number in (target_earliest_block + 1..=current_earliest).rev() {
            phase_totals.add(self.backfill_block_with_snapshot(block_number)?);

            let done = current_earliest - block_number + 1;
            let is_final = block_number == target_earliest_block + 1;
            if done.is_multiple_of(LOG_EVERY) || is_final {
                let elapsed_secs = start.elapsed().as_secs_f64();
                let blocks_per_sec =
                    if elapsed_secs.is_normal() { done as f64 / elapsed_secs } else { 0.0 };
                let eta_secs = if blocks_per_sec.is_normal() && blocks_per_sec > 0.0 {
                    (total - done) as f64 / blocks_per_sec
                } else {
                    0.0
                };
                let progress_pct = (done as f64 / total as f64) * 100.0;
                let avg = phase_totals.averages(done);
                let (from_changeset, from_current_state) =
                    crate::db::find_source_stats::snapshot_and_reset();
                let find_source_total = from_changeset + from_current_state;
                info!(
                    target: "reth::op-proofs::backfill",
                    strategy = "snapshot",
                    done,
                    total,
                    avg_reverts = ?avg.reverts,
                    avg_trie_changesets = ?avg.trie_changesets,
                    avg_prepend = ?avg.prepend,
                    avg_validate = ?avg.validate,
                    find_source_total,
                    "progress: {progress_pct:.2}% ({blocks_per_sec:.1} blk/s, ETA {eta_secs:.0}s)"
                );
            }
        }

        let final_avg = phase_totals.averages(total);
        info!(
            target: "reth::op-proofs::backfill",
            strategy = "snapshot",
            blocks = total,
            elapsed = ?start.elapsed(),
            avg_reverts = ?final_avg.reverts,
            avg_trie_changesets = ?final_avg.trie_changesets,
            avg_prepend = ?final_avg.prepend,
            avg_validate = ?final_avg.validate,
            "Proofs backfill complete"
        );

        Ok(())
    }

    /// One step of snapshot-strategy backfill: prepend block `N` and advance
    /// the snapshot to track `N-1`.
    fn backfill_block_with_snapshot(
        &self,
        block_number: BlockNumber,
    ) -> Result<PhaseTimings, BackfillError> {
        debug!(target: "reth::op-proofs::backfill", block_number, "snapshot-backfilling block");

        let block_hash = self
            .provider
            .block_hash(block_number)?
            .ok_or_else(|| ProviderError::HeaderNotFound(block_number.into()))?;
        let parent_hash = self
            .provider
            .block_hash(block_number - 1)?
            .ok_or_else(|| ProviderError::HeaderNotFound((block_number - 1).into()))?;

        // Compute diff using the snapshot (cheap trie cursor reads, no merge walk).
        let diff;
        let compute_t: ComputeTimings;
        {
            let proofs_ro = self.storage.provider_ro()?;
            (diff, compute_t) = compute_block_backfill_diff_with_snapshot(
                &self.provider,
                &proofs_ro,
                block_number,
            )?;
            debug!(
                target: "reth::op-proofs::backfill",
                block_number,
                reverts_elapsed = ?compute_t.reverts,
                trie_changesets_elapsed = ?compute_t.trie_changesets,
                accounts = diff.sorted_post_state.accounts.len(),
                storages = diff.sorted_post_state.storages.len(),
                account_trie_nodes = diff.sorted_trie_updates.account_nodes_ref().len(),
                storage_tries = diff.sorted_trie_updates.storage_tries_ref().len(),
                "computed backfill diff (snapshot)"
            );
        }

        let block_ref = BlockWithParent {
            block: BlockNumHash::new(block_number, block_hash),
            parent: parent_hash,
        };

        let prepend_start = Instant::now();
        let bp = self.storage.backfill_provider()?;

        // Move the snapshot to `(N-1, parent_hash)` BEFORE prepend_block so we
        // can pass `diff` by value to `prepend_block` without cloning. Both
        // operations land in the same rw-tx and commit atomically.
        bp.update_snapshot(
            BlockNumHash::new(block_number - 1, parent_hash),
            &diff.sorted_trie_updates,
        )?;

        let counts = bp.prepend_block(block_ref, diff)?;
        let prepend = prepend_start.elapsed();
        debug!(
            target: "reth::op-proofs::backfill",
            block_number,
            elapsed = ?prepend,
            ?counts,
            "prepend_block + snapshot revert done"
        );

        // Validate against the reth header. The snapshot now reflects
        // state-at-(N-1), so we use `SnapshotTrieCursorFactory` for trie reads.
        // The hashed-leaf cursors still go through the merge-walk at max=N-1.
        let validate_start = Instant::now();
        let expected_root = self
            .provider
            .header_by_number(block_number - 1)?
            .ok_or_else(|| ProviderError::HeaderNotFound((block_number - 1).into()))?
            .state_root();
        let computed_root = {
            use crate::{OpProofsHashedAccountCursorFactory, SnapshotTrieCursorFactory};
            use reth_trie::hashed_cursor::HashedPostStateCursorFactory;
            let state_sorted = HashedPostState::default().into_sorted();
            StateRoot::new(
                SnapshotTrieCursorFactory::new(&bp),
                HashedPostStateCursorFactory::new(
                    OpProofsHashedAccountCursorFactory::new(&bp, block_number - 1),
                    &state_sorted,
                ),
            )
            .root()?
        };
        let validate = validate_start.elapsed();
        debug!(
            target: "reth::op-proofs::backfill",
            block_number,
            elapsed = ?validate,
            "state root validation done (snapshot)"
        );
        if computed_root != expected_root {
            return Err(BackfillError::StateRootMismatch {
                block_number,
                computed: computed_root,
                expected: expected_root,
            });
        }

        OpProofsBackfillProvider::commit(bp)?;

        Ok(PhaseTimings {
            reverts: compute_t.reverts,
            trie_changesets: compute_t.trie_changesets,
            prepend,
            validate,
        })
    }
}
