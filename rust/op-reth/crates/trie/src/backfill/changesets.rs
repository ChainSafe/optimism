//! Per-block backfill diff computation.
//!
//! For block N, [`compute_block_backfill_diff`] returns:
//! - `HashedPostStateSorted` — per-block leaf revert (account & storage values before block N ran).
//!   Read directly from reth's `AccountChangeSets` / `StorageChangeSets` and reused as
//!   `BlockStateDiff::sorted_post_state`.
//! - `TrieUpdatesSorted` — trie-node before-values for paths block N touched. Written into the four
//!   changeset tables by `prepend_block`.
//!
//! # Algorithm
//!
//! Conceptually equivalent to
//! `reth_trie_db::changesets::compute_block_trie_changesets_inner`, but reads
//! the trie from the **op-reth proofs storage** at `max_block_number = N`
//! instead of from reth's current-state tables. That swap is what makes the
//! per-block cost scale with `k_N` (this block's diff) rather than `K_tail`
//! (every changeset entry between N and the DB tip).
//!
//! A single call to `overlay_root_from_nodes_with_updates` does the work:
//! - **Cursor**: `OpProofsHashedAccountCursorFactory` / `OpProofsTrieCursorFactory` at `max=N` —
//!   serves state@N.
//! - **State overlay**: the per-block leaf revert (`individual_state_revert`). Walked together with
//!   the cursor, this yields state@N-1 for every leaf block N touched.
//! - **Prefix sets**: only the paths block N's leaf revert covers.
//!
//! The returned `TrieUpdates` is the difference between trie@N (cursor view)
//! and trie@N-1 (after applying the overlay) for those paths — which is
//! exactly the trie changeset we want:
//! - branch modified at N → `(path, Some(value_at_N-1))`
//! - branch destroyed at N (existed at N-1, gone at N) → `(path, Some(value_at_N-1))`
//! - branch created at N (existed at N, gone at N-1) → `(path, None)` via `removed_nodes`

use crate::{
    BlockStateDiff, OpProofsHashedAccountCursorFactory, OpProofsProviderRO, OpProofsSnapshotReader,
    SnapshotTrieCursorFactory, backfill::error::BackfillError, proof::DatabaseStateRoot,
};
use alloy_primitives::BlockNumber;
use reth_provider::{
    BlockNumReader, ChangeSetReader, DBProvider, ProviderError, StorageChangeSetReader,
    StorageSettingsCache,
};
use reth_trie::{
    StateRoot, TrieInput, hashed_cursor::HashedPostStateCursorFactory,
    trie_cursor::InMemoryTrieCursorFactory,
};
use reth_trie_common::{HashedPostStateSorted, updates::TrieUpdatesSorted};
use reth_trie_db::from_reverts_auto;
use std::time::{Duration, Instant};

/// Wall-clock breakdown of [`compute_block_backfill_diff`]. Exposed so the
/// caller (`BackfillJob`) can aggregate sub-phase averages into its progress
/// log without instrumenting the internals separately.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct ComputeTimings {
    /// Time spent in `from_reverts_auto(N..=N)` — reading reth's per-block
    /// account and storage changesets to build the leaf revert.
    pub(super) reverts: Duration,
    /// Time spent in [`compute_trie_changesets_against_proofs`] — the
    /// proofs-storage `overlay_root_from_nodes_with_updates` walk.
    pub(super) trie_changesets: Duration,
}

/// Compute the [`BlockStateDiff`] for `block_number` — exactly what
/// [`crate::OpProofsBackfillProvider::prepend_block`] expects:
/// - `sorted_trie_updates`: trie-node before-values for paths block N touched.
/// - `sorted_post_state`: per-block leaf revert (account & storage values
///   before block N ran).
///
/// `proofs_provider` must reflect the proofs-storage state *at the start of
/// this iteration* — i.e. `earliest == block_number`. Callers should open a
/// fresh RO provider per iteration so it sees writes committed by the
/// previous `prepend_block`.
pub(super) fn compute_block_backfill_diff<P, R>(
    reth_provider: &P,
    proofs_provider: R,
    block_number: BlockNumber,
) -> Result<(BlockStateDiff, ComputeTimings), BackfillError>
where
    P: ChangeSetReader
        + StorageChangeSetReader
        + BlockNumReader
        + DBProvider
        + StorageSettingsCache,
    R: OpProofsProviderRO + Clone,
{
    // Per-block leaf revert: doubles as `post_state` for `prepend_block` and
    // as the state overlay for the trie@N-1 reconstruction below.
    let reverts_start = Instant::now();
    let sorted_post_state = from_reverts_auto(reth_provider, block_number..=block_number)?;
    let reverts = reverts_start.elapsed();

    let trie_changesets_start = Instant::now();
    let sorted_trie_updates = compute_trie_changesets_against_proofs(
        proofs_provider,
        block_number,
        &sorted_post_state,
    )?;
    let trie_changesets_elapsed = trie_changesets_start.elapsed();

    Ok((
        BlockStateDiff { sorted_trie_updates, sorted_post_state },
        ComputeTimings { reverts, trie_changesets: trie_changesets_elapsed },
    ))
}

fn compute_trie_changesets_against_proofs<R>(
    proofs_provider: R,
    block_number: BlockNumber,
    individual_state_revert: &HashedPostStateSorted,
) -> Result<TrieUpdatesSorted, BackfillError>
where
    R: OpProofsProviderRO + Clone,
{
    // Apply block N's leaf revert as a state overlay on top of the proofs
    // cursor at max=N, then walk just the paths block N touched. The returned
    // `TrieUpdates` describes how trie@N-1 differs from the cursor's view at
    // max=N — which is exactly the changeset:
    //   - modified branch  → (path, Some(value_at_N-1))
    //   - destroyed at N   → (path, Some(value_at_N-1))
    //   - created at N     → (path, None)   (via `removed_nodes`)
    let input = TrieInput {
        nodes: Default::default(),
        state: individual_state_revert.clone().into(),
        prefix_sets: individual_state_revert.construct_prefix_sets(),
    };
    let (_, trie_updates) =
        StateRoot::overlay_root_from_nodes_with_updates(proofs_provider, block_number, input)
            .map_err(ProviderError::other)?;
    Ok(trie_updates.into_sorted())
}

/// Snapshot-backed analog of [`compute_block_backfill_diff`].
///
/// Equivalent to [`compute_block_backfill_diff`] except the trie-changeset
/// computation reads from [`SnapshotTrieCursorFactory`] (a direct read of the
/// snapshot tables) instead of [`crate::OpProofsTrieCursorFactory`] (the
/// history-aware merge walk over `V2*Trie` + `V2*TrieHistory` +
/// `V2*TrieChangeSets`).
///
/// # Preconditions
///
/// The caller must guarantee that the snapshot reflects trie state at
/// `block_number` — i.e. [`crate::db::SnapshotMeta::earliest`] equals
/// `BlockNumHash::new(block_number, _)` and the status is `Ready`. The
/// `BackfillJob` checks this before dispatching to the snapshot path.
pub(super) fn compute_block_backfill_diff_with_snapshot<P, R>(
    reth_provider: &P,
    proofs_provider: R,
    block_number: BlockNumber,
) -> Result<(BlockStateDiff, ComputeTimings), BackfillError>
where
    P: ChangeSetReader
        + StorageChangeSetReader
        + BlockNumReader
        + DBProvider
        + StorageSettingsCache,
    R: OpProofsProviderRO + OpProofsSnapshotReader + Clone,
{
    let reverts_start = Instant::now();
    let sorted_post_state = from_reverts_auto(reth_provider, block_number..=block_number)?;
    let reverts = reverts_start.elapsed();

    let trie_changesets_start = Instant::now();
    let sorted_trie_updates = compute_trie_changesets_against_snapshot(
        proofs_provider,
        block_number,
        &sorted_post_state,
    )?;
    let trie_changesets_elapsed = trie_changesets_start.elapsed();

    Ok((
        BlockStateDiff { sorted_trie_updates, sorted_post_state },
        ComputeTimings { reverts, trie_changesets: trie_changesets_elapsed },
    ))
}

fn compute_trie_changesets_against_snapshot<R>(
    proofs_provider: R,
    block_number: BlockNumber,
    individual_state_revert: &HashedPostStateSorted,
) -> Result<TrieUpdatesSorted, BackfillError>
where
    R: OpProofsProviderRO + OpProofsSnapshotReader + Clone,
{
    // Same algorithm as `compute_trie_changesets_against_proofs`, but with the
    // trie cursor swapped for the snapshot reader. The snapshot already reflects
    // trie state at `block_number`, so no per-node `find_source` work is needed.
    //
    // The hashed-leaf cursors still come from `proofs_provider` at max=N. The
    // leaf walk is sparse here (only paths in the prefix set are visited), so
    // its cost stays bounded by the per-block diff size.
    let nodes_sorted = TrieInput::default().nodes.into_sorted();
    let prefix_sets = individual_state_revert.construct_prefix_sets().freeze();
    let (_, trie_updates) = StateRoot::new(
        InMemoryTrieCursorFactory::new(
            SnapshotTrieCursorFactory::new(proofs_provider.clone()),
            &nodes_sorted,
        ),
        HashedPostStateCursorFactory::new(
            OpProofsHashedAccountCursorFactory::new(proofs_provider, block_number),
            individual_state_revert,
        ),
    )
    .with_prefix_sets(prefix_sets)
    .root_with_updates()
    .map_err(ProviderError::other)?;
    Ok(trie_updates.into_sorted())
}
