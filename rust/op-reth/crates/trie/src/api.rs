//! Storage API for external storage of intermediary trie nodes.

use crate::{
    OpProofsStorageResult,
    db::{HashedStorageKey, SnapshotMeta, StorageTrieKey},
};
use alloy_eips::{BlockNumHash, eip1898::BlockWithParent};
use alloy_primitives::{B256, U256};
use auto_impl::auto_impl;
use derive_more::{AddAssign, Constructor};
use reth_primitives_traits::Account;
use reth_trie::{
    hashed_cursor::{HashedCursor, HashedStorageCursor},
    trie_cursor::{TrieCursor, TrieStorageCursor},
};
use reth_trie_common::{
    BranchNodeCompact, HashedPostStateSorted, Nibbles, StoredNibbles, StoredNibblesSubKey,
    updates::TrieUpdatesSorted,
};
use std::{fmt::Debug, time::Duration};

/// Duration metrics for block processing.
#[derive(Debug, Default, Clone)]
pub struct OperationDurations {
    /// Total time to process a block (end-to-end) in seconds
    pub total_duration_seconds: Duration,
    /// Time spent executing the block (EVM) in seconds
    pub execution_duration_seconds: Duration,
    /// Time spent calculating state root in seconds
    pub state_root_duration_seconds: Duration,
    /// Time spent writing trie updates to storage in seconds
    pub write_duration_seconds: Duration,
}

/// Diff of trie updates and post state for a block.
#[derive(Debug, Clone, Default)]
pub struct BlockStateDiff {
    /// Trie updates for branch nodes
    pub sorted_trie_updates: TrieUpdatesSorted,
    /// Post state for leaf nodes (accounts and storage)
    pub sorted_post_state: HashedPostStateSorted,
}

impl BlockStateDiff {
    /// Extend the [` BlockStateDiff`] from other latest [`BlockStateDiff`]
    pub fn extend_ref(&mut self, other: &Self) {
        self.sorted_trie_updates.extend_ref_and_sort(&other.sorted_trie_updates);
        self.sorted_post_state.extend_ref_and_sort(&other.sorted_post_state);
    }
}

/// Counts of trie updates written to storage.
#[derive(Debug, Clone, Default, AddAssign, Constructor, Eq, PartialEq)]
pub struct WriteCounts {
    /// Number of account trie updates written
    pub account_trie_updates_written_total: u64,
    /// Number of storage trie updates written
    pub storage_trie_updates_written_total: u64,
    /// Number of hashed accounts written
    pub hashed_accounts_written_total: u64,
    /// Number of hashed storages written
    pub hashed_storages_written_total: u64,
}

/// Provider for interacting with the proofs storage within a transaction.
#[auto_impl(Arc)]
pub trait OpProofsProviderRO: Send + Sync + Debug {
    /// Cursor for iterating over trie branches.
    type StorageTrieCursor<'tx>: TrieStorageCursor + 'tx
    where
        Self: 'tx;

    /// Cursor for iterating over account trie branches.
    type AccountTrieCursor<'tx>: TrieCursor + 'tx
    where
        Self: 'tx;

    /// Cursor for iterating over storage leaves.
    type StorageCursor<'tx>: HashedStorageCursor<Value = U256> + Send + 'tx
    where
        Self: 'tx;

    /// Cursor for iterating over account leaves.
    type AccountHashedCursor<'tx>: HashedCursor<Value = Account> + Send + 'tx
    where
        Self: 'tx;

    /// Get the earliest block number and hash that has been stored
    fn get_earliest_block_number(&self) -> OpProofsStorageResult<Option<(u64, B256)>>;

    /// Get the latest block number and hash that has been stored
    fn get_latest_block_number(&self) -> OpProofsStorageResult<Option<(u64, B256)>>;

    /// Get a trie cursor for the storage backend
    fn storage_trie_cursor<'tx>(
        &self,
        hashed_address: B256,
        max_block_number: u64,
    ) -> OpProofsStorageResult<Self::StorageTrieCursor<'tx>>;

    /// Get a trie cursor for the account backend
    fn account_trie_cursor<'tx>(
        &self,
        max_block_number: u64,
    ) -> OpProofsStorageResult<Self::AccountTrieCursor<'tx>>;

    /// Get a storage cursor for the storage backend
    fn storage_hashed_cursor<'tx>(
        &self,
        hashed_address: B256,
        max_block_number: u64,
    ) -> OpProofsStorageResult<Self::StorageCursor<'tx>>;

    /// Get an account hashed cursor for the storage backend
    fn account_hashed_cursor<'tx>(
        &self,
        max_block_number: u64,
    ) -> OpProofsStorageResult<Self::AccountHashedCursor<'tx>>;

    /// Fetch all updates for a given block number.
    fn fetch_trie_updates(&self, block_number: u64) -> OpProofsStorageResult<BlockStateDiff>;
}

/// Provider for writing to the proofs storage within a transaction.
pub trait OpProofsProviderRw: OpProofsProviderRO {
    /// Store trie updates for a block.
    fn store_trie_updates(
        &self,
        block_ref: BlockWithParent,
        block_state_diff: BlockStateDiff,
    ) -> OpProofsStorageResult<WriteCounts>;

    /// Store a batch of trie updates for a block.
    fn store_trie_updates_batch(
        &self,
        updates: Vec<(BlockWithParent, BlockStateDiff)>,
    ) -> OpProofsStorageResult<WriteCounts>;

    /// Applies [`BlockStateDiff`] to the earliest state (updating/deleting nodes) and updates the
    /// earliest block number.
    fn prune_earliest_state(
        &self,
        new_earliest_block_ref: BlockWithParent,
    ) -> OpProofsStorageResult<WriteCounts>;

    /// Remove account, storage and trie updates from historical storage for all blocks till
    /// the specified block (inclusive).
    fn unwind_history(&self, to: BlockWithParent) -> OpProofsStorageResult<()>;

    /// Deletes all updates > `latest_common_block` and replaces them with the new updates.
    fn replace_updates(
        &self,
        latest_common_block: BlockNumHash,
        blocks_to_add: Vec<(BlockWithParent, BlockStateDiff)>,
    ) -> OpProofsStorageResult<()>;

    /// Set the earliest block number and hash that has been stored
    fn set_earliest_block_number(&self, block_number: u64, hash: B256)
    -> OpProofsStorageResult<()>;

    /// Commit the changes to the database.
    /// Consumes the provider.
    fn commit(self) -> OpProofsStorageResult<()>;
}

/// Provider for writing historical records for blocks older than the current window boundary.
///
/// Unlike [`OpProofsProviderRw::store_trie_updates`], which is strictly append-only (validates
/// parent hash against `latest` and advances `latest`), this provider is designed for
/// **prepend-style** writes that extend the window backward.  It does not touch the `latest`
/// marker, and it does not enforce parent-hash ordering against `latest`.
///
/// The typical call sequence for one backfill step is:
/// ```ignore
/// let bp = storage.backfill_provider()?;
/// bp.prepend_block(block_ref, diff)?;
/// bp.commit()?;
/// ```
pub trait OpProofsBackfillProvider: OpProofsProviderRO {
    /// Write historical changeset and history-bitmap entries for `block_ref`, and move the
    /// `earliest` marker to `block_ref.parent`.
    ///
    /// `diff` contains:
    /// - `sorted_trie_updates`: trie node **before-values** for `block_ref.block.number` (i.e. what
    ///   each changed node looked like *before* the block executed).
    /// - `sorted_post_state`: account / storage **before-values** for the same block.
    ///
    /// The implementation must **not** update the `latest` marker and must **not**
    /// validate `diff` against the current `latest` block.
    fn prepend_block(
        &self,
        block_ref: BlockWithParent,
        diff: BlockStateDiff,
    ) -> OpProofsStorageResult<WriteCounts>;

    /// Commit the transaction. Consumes the provider.
    fn commit(self) -> OpProofsStorageResult<()>;
}

/// Blanket impl of [`OpProofsProviderRO`] for shared references.
///
/// This allows passing `&bp` (where `bp: OpProofsBackfillProvider + OpProofsProviderRO`)
/// to APIs that require `P: OpProofsProviderRO + Clone`. Since `&T: Copy`, cloning a
/// reference is free, enabling `StateRoot::overlay_root(&bp, ...)` to work without
/// requiring the underlying provider to implement `Clone`.
impl<'a, T: OpProofsProviderRO + 'a> OpProofsProviderRO for &'a T {
    type StorageTrieCursor<'tx>
        = T::StorageTrieCursor<'tx>
    where
        Self: 'tx,
        T: 'tx;
    type AccountTrieCursor<'tx>
        = T::AccountTrieCursor<'tx>
    where
        Self: 'tx,
        T: 'tx;
    type StorageCursor<'tx>
        = T::StorageCursor<'tx>
    where
        Self: 'tx,
        T: 'tx;
    type AccountHashedCursor<'tx>
        = T::AccountHashedCursor<'tx>
    where
        Self: 'tx,
        T: 'tx;

    fn get_earliest_block_number(&self) -> crate::OpProofsStorageResult<Option<(u64, B256)>> {
        T::get_earliest_block_number(self)
    }

    fn get_latest_block_number(&self) -> crate::OpProofsStorageResult<Option<(u64, B256)>> {
        T::get_latest_block_number(self)
    }

    fn storage_trie_cursor<'tx>(
        &self,
        hashed_address: B256,
        max_block_number: u64,
    ) -> crate::OpProofsStorageResult<Self::StorageTrieCursor<'tx>>
    where
        'a: 'tx,
    {
        T::storage_trie_cursor(self, hashed_address, max_block_number)
    }

    fn account_trie_cursor<'tx>(
        &self,
        max_block_number: u64,
    ) -> crate::OpProofsStorageResult<Self::AccountTrieCursor<'tx>>
    where
        'a: 'tx,
    {
        T::account_trie_cursor(self, max_block_number)
    }

    fn storage_hashed_cursor<'tx>(
        &self,
        hashed_address: B256,
        max_block_number: u64,
    ) -> crate::OpProofsStorageResult<Self::StorageCursor<'tx>>
    where
        'a: 'tx,
    {
        T::storage_hashed_cursor(self, hashed_address, max_block_number)
    }

    fn account_hashed_cursor<'tx>(
        &self,
        max_block_number: u64,
    ) -> crate::OpProofsStorageResult<Self::AccountHashedCursor<'tx>>
    where
        'a: 'tx,
    {
        T::account_hashed_cursor(self, max_block_number)
    }

    fn fetch_trie_updates(
        &self,
        block_number: u64,
    ) -> crate::OpProofsStorageResult<BlockStateDiff> {
        T::fetch_trie_updates(self, block_number)
    }
}

/// Read access to the optional trie-state snapshot used to accelerate backfills.
///
/// Implemented by backends that maintain a snapshot of trie state at the moving
/// `earliest` boundary (currently only the v2 MDBX backend — see
/// [`crate::db::store_v2`]). Backends that do not maintain a snapshot return
/// [`None`] from [`Self::snapshot_meta`]; callers must treat that as "no
/// snapshot available" and fall back to the merge-walk path.
///
/// The cursors returned by [`Self::snapshot_account_trie_cursor`] /
/// [`Self::snapshot_storage_trie_cursor`] read directly from the snapshot
/// tables — they perform no history bitmap lookups. They are only valid to use
/// when [`Self::snapshot_meta`] returns
/// `Some(meta)` with `meta.status ==`
/// [`SnapshotStatus::Ready`](crate::db::SnapshotStatus::Ready) **and** the
/// caller's target block equals `meta.earliest.number`.
#[auto_impl(Arc)]
pub trait OpProofsSnapshotReader: Send + Sync + Debug {
    /// Cursor over the snapshot's account trie table.
    type SnapshotAccountTrieCursor<'tx>: TrieCursor + 'tx
    where
        Self: 'tx;

    /// Cursor over the snapshot's storage trie table.
    type SnapshotStorageTrieCursor<'tx>: TrieStorageCursor + 'tx
    where
        Self: 'tx;

    /// Read the snapshot metadata, if a snapshot exists.
    fn snapshot_meta(&self) -> OpProofsStorageResult<Option<SnapshotMeta>>;

    /// Open a cursor over the snapshot's account trie table.
    fn snapshot_account_trie_cursor<'tx>(
        &self,
    ) -> OpProofsStorageResult<Self::SnapshotAccountTrieCursor<'tx>>;

    /// Open a cursor over the snapshot's storage trie table for `hashed_address`.
    fn snapshot_storage_trie_cursor<'tx>(
        &self,
        hashed_address: B256,
    ) -> OpProofsStorageResult<Self::SnapshotStorageTrieCursor<'tx>>;
}

/// Blanket impl of [`OpProofsSnapshotReader`] for shared references.
///
/// Mirrors the [`OpProofsProviderRO`] blanket so a `&P` can be threaded into
/// the same APIs that already accept `&P` for the merge-walk reader.
impl<'a, T: OpProofsSnapshotReader + 'a> OpProofsSnapshotReader for &'a T {
    type SnapshotAccountTrieCursor<'tx>
        = T::SnapshotAccountTrieCursor<'tx>
    where
        Self: 'tx,
        T: 'tx;
    type SnapshotStorageTrieCursor<'tx>
        = T::SnapshotStorageTrieCursor<'tx>
    where
        Self: 'tx,
        T: 'tx;

    fn snapshot_meta(&self) -> OpProofsStorageResult<Option<SnapshotMeta>> {
        T::snapshot_meta(self)
    }

    fn snapshot_account_trie_cursor<'tx>(
        &self,
    ) -> OpProofsStorageResult<Self::SnapshotAccountTrieCursor<'tx>>
    where
        'a: 'tx,
    {
        T::snapshot_account_trie_cursor(self)
    }

    fn snapshot_storage_trie_cursor<'tx>(
        &self,
        hashed_address: B256,
    ) -> OpProofsStorageResult<Self::SnapshotStorageTrieCursor<'tx>>
    where
        'a: 'tx,
    {
        T::snapshot_storage_trie_cursor(self, hashed_address)
    }
}

/// Write access to the optional trie-state snapshot used to accelerate backfills.
///
/// Implemented by writer providers on backends that maintain a snapshot (currently
/// only the v2 MDBX backend — see [`crate::db::store_v2`]). All operations share
/// the underlying transaction with whichever other writer trait the same provider
/// implements (e.g. [`OpProofsBackfillProvider`]), so snapshot mutations commit
/// atomically with the changeset / history writes that accompany them.
///
/// The lifecycle states encoded in [`SnapshotMeta`] are managed by the caller:
///
/// 1. [`SnapshotInitJob`](crate::backfill) populates the snapshot tables, then
///    [`Self::set_snapshot_meta`] flips the status from `Building` to `Ready`.
/// 2. Each backfill iteration calls [`Self::apply_snapshot_revert`] to advance
///    the snapshot one block backward, then [`Self::set_snapshot_meta`] to
///    record the new `earliest`.
/// 3. If the snapshot falls out of sync with the proofs window, the caller
///    flips status to `Stale` (or calls [`Self::clear_snapshot`] to drop it).
pub trait OpProofsSnapshotProvider: OpProofsSnapshotReader {
    /// Overwrite the singleton snapshot meta row.
    fn set_snapshot_meta(&self, meta: SnapshotMeta) -> OpProofsStorageResult<()>;

    /// Apply a single block's trie reverts to the snapshot, advancing it one
    /// block backward.
    ///
    /// `trie_updates` carries the **before-values** for the block being
    /// reverted (the same payload prepended into `V2*TrieChangeSets`):
    /// - `(path, Some(node))` → restore `snapshot[path] = node`
    /// - `(path, None)` → remove `path` from the snapshot (it did not exist
    ///   before that block)
    ///
    /// Does **not** update [`SnapshotMeta::earliest`]; the caller must invoke
    /// [`Self::set_snapshot_meta`] to record the new boundary atomically within
    /// the same transaction.
    fn apply_snapshot_revert(
        &self,
        trie_updates: &TrieUpdatesSorted,
    ) -> OpProofsStorageResult<u64>;

    /// Commit the transaction. Consumes the provider.
    fn commit(self) -> OpProofsStorageResult<()>;
}

/// Anchor describing the state of an in-progress or completed snapshot init.
///
/// Mirrors [`InitialStateAnchor`]'s shape: combines the lifecycle status from
/// [`SnapshotMeta`] with the resume keys recovered from the destination
/// snapshot tables. Used by the snapshot-init job to decide whether to start
/// fresh, resume, or refuse.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SnapshotInitAnchor {
    /// Current snapshot meta, or `None` if no snapshot row has been written.
    pub meta: Option<SnapshotMeta>,
    /// Last key in [`crate::db::V2AccountsTrieSnapshot`]; `None` if the table
    /// is empty. Used to resume the account-trie phase past whatever was last
    /// committed.
    pub last_account_trie_key: Option<StoredNibbles>,
    /// Last `(hashed_address, subkey)` pair in [`crate::db::V2StoragesTrieSnapshot`];
    /// `None` if the table is empty. Used to resume the storage-trie phase.
    pub last_storage_trie_key: Option<(B256, StoredNibblesSubKey)>,
}

/// Provider for building the trie-state snapshot.
///
/// Mirrors [`OpProofsInitProvider`]'s role for the original proofs init:
/// a single trait that bundles all the read + write operations a snapshot-init
/// job needs, isolated from the per-iteration backfill writer
/// ([`OpProofsSnapshotProvider`]).
///
/// The job (see [`crate::snapshot::SnapshotInitJob`]) drives this trait in a
/// loop of short rw-transactions:
///
/// 1. Read [`Self::snapshot_init_anchor`] once at start to classify state
///    (fresh / resume / refuse).
/// 2. Per chunk: call [`Self::account_trie_source_chunk`] (or storage variant)
///    on an RO provider, then [`Self::store_account_trie_snapshot_branches`]
///    (or storage variant) on a fresh RW provider, then [`Self::commit`].
/// 3. After both phases drain, transition status via [`Self::set_snapshot_meta`]
///    to [`SnapshotStatus::Ready`](crate::db::SnapshotStatus::Ready).
///
/// On a crash mid-init, meta stays at [`SnapshotStatus::Building`](crate::db::SnapshotStatus::Building);
/// a re-run inspects [`Self::snapshot_init_anchor`], discovers the resume keys
/// from the partially-populated destination tables, and continues from there.
pub trait OpProofsSnapshotInitProvider: Send + Sync + Debug {
    /// Read the snapshot init anchor (meta + resume keys) in one call.
    fn snapshot_init_anchor(&self) -> OpProofsStorageResult<SnapshotInitAnchor>;

    /// Overwrite the singleton snapshot meta row.
    fn set_snapshot_meta(&self, meta: SnapshotMeta) -> OpProofsStorageResult<()>;

    /// Wipe both snapshot tables and the meta row. Used to rebuild from
    /// scratch or to drop a stale snapshot.
    fn clear_snapshot(&self) -> OpProofsStorageResult<()>;

    /// Append a chunk of account-trie entries to [`crate::db::V2AccountsTrieSnapshot`].
    ///
    /// Entries **must** be in ascending key order **and** strictly greater than
    /// the snapshot table's current last key. The init job streams them from
    /// the source cursor in sorted order, so the invariant holds naturally.
    /// Internally uses `append` (no in-place updates).
    fn store_account_trie_snapshot_branches(
        &self,
        entries: Vec<(StoredNibbles, BranchNodeCompact)>,
    ) -> OpProofsStorageResult<()>;

    /// Append a chunk of storage-trie entries to [`crate::db::V2StoragesTrieSnapshot`].
    ///
    /// Entries **must** be in ascending `(hashed_address, subkey)` order and
    /// strictly greater than the snapshot table's current last entry. Uses
    /// `append_dup`.
    fn store_storage_trie_snapshot_branches(
        &self,
        entries: Vec<(B256, StoredNibblesSubKey, BranchNodeCompact)>,
    ) -> OpProofsStorageResult<()>;

    /// Read up to `max_entries` rows from the source account-trie table
    /// (`V2AccountsTrie`) in ascending key order, **strictly greater than**
    /// `resume_after`.
    fn account_trie_source_chunk(
        &self,
        resume_after: Option<StoredNibbles>,
        max_entries: usize,
    ) -> OpProofsStorageResult<Vec<(StoredNibbles, BranchNodeCompact)>>;

    /// Read up to `max_entries` rows from the source storage-trie table
    /// (`V2StoragesTrie`) in ascending `(hashed_address, subkey)` order,
    /// **strictly greater than** `resume_after`.
    fn storage_trie_source_chunk(
        &self,
        resume_after: Option<(B256, StoredNibblesSubKey)>,
        max_entries: usize,
    ) -> OpProofsStorageResult<Vec<(B256, StoredNibblesSubKey, BranchNodeCompact)>>;

    /// Commit the transaction. Consumes the provider.
    fn commit(self) -> OpProofsStorageResult<()>;
}

/// Factory trait for creating providers to interact with the proofs storage.
#[auto_impl(Arc)]
pub trait OpProofsStore: Send + Sync + Debug {
    /// The read-only provider type created by the factory.
    type ProviderRO<'a>: OpProofsProviderRO + Clone + 'a
    where
        Self: 'a;

    /// The read-write provider type created by the factory.
    type ProviderRw<'a>: OpProofsProviderRw + 'a
    where
        Self: 'a;

    /// The initialization provider type created by the factory.
    type Initializer<'a>: OpProofsInitProvider + 'a
    where
        Self: 'a;

    /// The backfill provider type created by the factory.
    type BackfillProvider<'a>: OpProofsBackfillProvider + 'a
    where
        Self: 'a;

    /// Create a read-only provider for interacting with the proofs storage.
    fn provider_ro<'a>(&'a self) -> OpProofsStorageResult<Self::ProviderRO<'a>>;

    /// Create a read-write provider for interacting with the proofs storage.
    fn provider_rw<'a>(&'a self) -> OpProofsStorageResult<Self::ProviderRw<'a>>;

    /// Create an initialization provider for interacting with the proofs storage.
    fn initialization_provider<'a>(&'a self) -> OpProofsStorageResult<Self::Initializer<'a>>;

    /// Create a backfill provider for prepend-style writes that extend the window backward.
    fn backfill_provider<'a>(&'a self) -> OpProofsStorageResult<Self::BackfillProvider<'a>>;
}

/// Status of the initial state anchor.
#[derive(Debug, Clone, Copy, Default)]
pub enum InitialStateStatus {
    /// Init isn't yet started
    #[default]
    NotStarted,
    /// Init is in progress (some tables may already be populated)
    InProgress,
    /// Init completed successfully (all tables done + earliest block set)
    Completed,
}

/// Anchor for the initial state.
#[derive(Debug, Clone, Default)]
pub struct InitialStateAnchor {
    /// The block for which the initial state is being initialized. None if initialization is not
    /// yet started.
    pub block: Option<BlockNumHash>,
    /// Whether initialization is still running or completed.
    pub status: InitialStateStatus,
    /// The latest key stored for `AccountTrieHistory`.
    pub latest_account_trie_key: Option<StoredNibbles>,
    /// The latest key stored for `StorageTrieHistory`.
    pub latest_storage_trie_key: Option<StorageTrieKey>,
    /// The latest key stored for `HashedAccountHistory`.
    pub latest_hashed_account_key: Option<B256>,
    /// The latest key stored for `HashedStorageHistory`.
    pub latest_hashed_storage_key: Option<HashedStorageKey>,
}

/// Trait for storing and retrieving the initial state anchor.
pub trait OpProofsInitProvider: Send + Sync + Debug {
    /// Read the current anchor.
    fn initial_state_anchor(&self) -> OpProofsStorageResult<InitialStateAnchor>;

    /// Create the anchor if it doesn't exist.
    /// Returns `Err` if an anchor already exists (prevents accidental overwrite).
    fn set_initial_state_anchor(&self, anchor: BlockNumHash) -> OpProofsStorageResult<()>;

    /// Store a batch of account trie branches. Used for saving existing state. For live state
    /// capture, use [`store_trie_updates`](OpProofsProviderRw::store_trie_updates).
    fn store_account_branches(
        &self,
        account_nodes: Vec<(Nibbles, Option<BranchNodeCompact>)>,
    ) -> OpProofsStorageResult<()>;

    /// Store a batch of storage trie branches. Used for saving existing state.
    fn store_storage_branches(
        &self,
        hashed_address: B256,
        storage_nodes: Vec<(Nibbles, Option<BranchNodeCompact>)>,
    ) -> OpProofsStorageResult<()>;

    /// Store a batch of account trie leaf nodes. Used for saving existing state.
    fn store_hashed_accounts(
        &self,
        accounts: Vec<(B256, Option<Account>)>,
    ) -> OpProofsStorageResult<()>;

    /// Store a batch of storage trie leaf nodes. Used for saving existing state.
    fn store_hashed_storages(
        &self,
        hashed_address: B256,
        storages: Vec<(B256, U256)>,
    ) -> OpProofsStorageResult<()>;

    /// Commit the initial state - mark the anchor as completed and also set the earliest block
    /// number to anchor.
    fn commit_initial_state(&self) -> OpProofsStorageResult<BlockNumHash>;

    /// Commit the changes to the database.
    /// Consumes the provider.
    fn commit(self) -> OpProofsStorageResult<()>;
}
