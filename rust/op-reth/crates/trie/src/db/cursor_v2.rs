//! V2 cursor implementations for the v2 table schema.
//!
//! These cursors implement **history-aware reads** using the v2 3-table-per-data-type pattern:
//!
//! | Purpose | Accounts | Storages | Account Trie | Storage Trie |
//! |---------|----------|----------|-------------|-------------|
//! | Current state | [`V2HashedAccounts`] | [`V2HashedStorages`] | [`V2AccountsTrie`] | [`V2StoragesTrie`] |
//! | `ChangeSets` | [`V2HashedAccountChangeSets`] | [`V2HashedStorageChangeSets`] | [`V2AccountTrieChangeSets`] | [`V2StorageTrieChangeSets`] |
//! | History | [`V2HashedAccountsHistory`] | [`V2HashedStoragesHistory`] | [`V2AccountsTrieHistory`] | [`V2StoragesTrieHistory`] |
//!
//! # Historical Lookup Strategy
//!
//! Each cursor accepts a `max_block_number` parameter. For each key encountered:
//!
//! 1. **History bitmap lookup**: Seek `ShardedKey(key, max_block_number)` in the history table.
//!    The bitmap tells us which blocks modified this key.
//! 2. **Find the first modification *after* `max_block_number`**: Using `rank` + `select` on
//!    the bitmap. `rank(max_block_number)` counts entries ≤ the target block;
//!    `select(rank)` returns the first entry strictly greater.
//! 3. **Determine where the value lives**:
//!    - If a block `> max_block_number` modified this key → read the **changeset** at that
//!      block. The changeset stores the value *before* that block's execution, which is
//!      the value at the end of `max_block_number`.
//!    - If no block after `max_block_number` modified this key → the **current state** table
//!      already has the correct value.
//!

use alloy_primitives::{B256, U256};
use reth_db::{
    cursor::{DbCursorRO, DbDupCursorRO},
    models::sharded_key::ShardedKey,
    table::Table,
    BlockNumberList, DatabaseError,
};
use reth_primitives_traits::Account;
use reth_trie::{
    hashed_cursor::{HashedCursor, HashedStorageCursor},
    trie_cursor::{TrieCursor, TrieStorageCursor},
    BranchNodeCompact, Nibbles, StorageTrieEntry, StoredNibbles, StoredNibblesSubKey,
};

use crate::db::{
    models::{
        AccountTrieShardedKey, V2AccountsTrie, V2AccountTrieChangeSets, V2AccountsTrieHistory,
        BlockNumberHashedAddress, V2HashedAccountChangeSets, HashedAccountShardedKey,
        V2HashedAccounts, V2HashedAccountsHistory, V2HashedStorageChangeSets,
        HashedStorageShardedKey, V2HashedStorages, V2HashedStoragesHistory, StorageTrieShardedKey,
        V2StoragesTrie, V2StorageTrieChangeSets, V2StoragesTrieHistory,
    },
};

/// Enum to define where to read the value for a given key at a specific block.
#[derive(Debug, Eq, PartialEq)]
enum ResolvedSource {
    /// Read the "before" value from the changeset at this block.
    /// The changeset stores the value *before* this block's execution,
    /// which equals the value at the end of `max_block_number`.
    FromChangeset(u64),
    /// No modification after the target block → current state has the value.
    FromCurrentState,
}

/// Search history bitmaps to determine where to read the value for a key
/// at a given `max_block_number`.
///
/// The algorithm:
/// 1. Seek the first history shard with `highest_block_number >= max_block_number`.
/// 2. Within that shard, find the first block strictly `> max_block_number`.
/// 3. If found → `FromChangeset(block)`.
/// 4. If the shard boundary was hit (all entries ≤ `max_block_number`), advance
///    to the next shard for the same key. If found → use its first entry.
/// 5. Otherwise → `FromCurrentState`.
fn find_source<T, C>(
    cursor: &mut C,
    seek_key: T::Key,
    max_block_number: u64,
    key_filter: impl Fn(&T::Key) -> bool,
) -> Result<ResolvedSource, DatabaseError>
where
    T: Table<Value = BlockNumberList>,
    C: DbCursorRO<T>,
{
    // 1. Seek the first shard with highest_block_number >= max_block_number.
    let shard = cursor.seek(seek_key)?.filter(|(k, _)| key_filter(k));

    let Some((_, chunk)) = shard else {
        // No history shard found for this key (or all shards have
        // highest < max_block_number). Current state is authoritative.
        return Ok(ResolvedSource::FromCurrentState);
    };

    // 2. rank(n) = count of entries ≤ n. select(rank) = first entry > n.
    let rank = chunk.rank(max_block_number);
    if let Some(block) = chunk.select(rank) {
        return Ok(ResolvedSource::FromChangeset(block));
    }

    // 3. All entries in this shard are ≤ max_block_number (shard boundary hit).
    //    The next shard (if it exists for the same key) starts after this one.
    if let Some((_, next_chunk)) = cursor.next()?.filter(|(k, _)| key_filter(k))
        && let Some(block) = next_chunk.select(0) {
            return Ok(ResolvedSource::FromChangeset(block));
        }

    Ok(ResolvedSource::FromCurrentState)
}

/// History-aware cursor over the [`V2HashedAccounts`] v2 tables.
///
/// Uses a **dual-cursor merge** to discover all account keys that existed at
/// `max_block_number`. This is necessary because an account deleted *after*
/// the target block no longer exists in the current-state table and would be
/// missed by a walk of current state alone. The merge walks both the
/// current-state cursor and the history-bitmap cursor in sorted order,
/// yielding the minimum key from each, resolving its value at the target
/// block, and skipping keys that did not exist at that block.
#[derive(Debug)]
pub struct V2AccountCursor<C, HC, CC> {
    /// Current state walk cursor.
    cursor: C,
    /// History bitmap cursor for resolving individual keys.
    history_cursor: HC,
    /// History bitmap cursor for merge-walking deleted keys.
    history_walk_cursor: HC,
    /// Changeset cursor.
    changeset_cursor: CC,
    /// Target block number for historical reads.
    max_block_number: u64,
    /// Pre-fetched next entry from the current state walk.
    cs_next: Option<(B256, Account)>,
    /// Pre-fetched next unique key from the history walk.
    hist_next_key: Option<B256>,
    /// Whether `seek` has been called to initialize the merge cursors.
    seeked: bool,
    /// Fast path: when `true`, skip all history/changeset lookups and
    /// read directly from the current-state table.
    is_latest: bool,
}

impl<C, HC, CC> V2AccountCursor<C, HC, CC> {
    /// Create a new [`V2AccountCursor`].
    pub const fn new(
        cursor: C,
        history_cursor: HC,
        history_walk_cursor: HC,
        changeset_cursor: CC,
        max_block_number: u64,
        is_latest: bool,
    ) -> Self {
        Self {
            cursor,
            history_cursor,
            history_walk_cursor,
            changeset_cursor,
            max_block_number,
            cs_next: None,
            hist_next_key: None,
            seeked: false,
            is_latest,
        }
    }
}

impl<C, HC, CC> V2AccountCursor<C, HC, CC>
where
    C: DbCursorRO<V2HashedAccounts>,
    HC: DbCursorRO<V2HashedAccountsHistory>,
    CC: DbCursorRO<V2HashedAccountChangeSets> + DbDupCursorRO<V2HashedAccountChangeSets>,
{
    /// Resolve an account using a pre-fetched current-state value.
    ///
    /// Does **not** touch the walk cursor, so it is safe to call from the
    /// merge loop (`find_next_live`).
    fn resolve_account_merge(
        &mut self,
        hashed_address: B256,
        cs_value: Option<&Account>,
    ) -> Result<Option<Account>, DatabaseError> {
        let history_key = HashedAccountShardedKey::new(hashed_address, self.max_block_number);
        let source = find_source::<V2HashedAccountsHistory, _>(
            &mut self.history_cursor,
            history_key,
            self.max_block_number,
            |k| k.0.key == hashed_address,
        )?;

        match source {
            ResolvedSource::FromChangeset(changeset_block) => {
                let entry = self
                    .changeset_cursor
                    .seek_by_key_subkey(changeset_block, hashed_address)?
                    .filter(|e| e.hashed_address == hashed_address);
                Ok(entry.and_then(|e| e.info))
            }
            ResolvedSource::FromCurrentState => Ok(cs_value.copied()),
        }
    }

    /// Advance the history walk cursor past all shards of `key` and return
    /// the next distinct key, if any.
    fn advance_history_past(
        &mut self,
        key: &B256,
    ) -> Result<Option<B256>, DatabaseError> {
        let entry =
            self.history_walk_cursor.seek(HashedAccountShardedKey::new(*key, u64::MAX))?;
        match entry {
            Some((k, _)) if k.0.key == *key => {
                // On the last shard of this key — one more step.
                Ok(self.history_walk_cursor.next()?.map(|(k, _)| k.0.key))
            }
            Some((k, _)) => Ok(Some(k.0.key)),
            None => Ok(None),
        }
    }

    /// Merge-walk both the current-state cursor and the history-bitmap cursor,
    /// yielding the next key (in ascending order) whose account is live at
    /// `max_block_number`.
    fn find_next_live(
        &mut self,
    ) -> Result<Option<(B256, Account)>, DatabaseError> {
        loop {
            let (min_key, cs_value) = match (&self.cs_next, &self.hist_next_key) {
                (Some((cs_k, cs_v)), Some(h_k)) => {
                    if cs_k <= h_k {
                        (*cs_k, Some(*cs_v))
                    } else {
                        (*h_k, None)
                    }
                }
                (Some((cs_k, cs_v)), None) => (*cs_k, Some(*cs_v)),
                (None, Some(h_k)) => (*h_k, None),
                (None, None) => return Ok(None),
            };

            // Advance whichever cursor(s) produced this key.
            if self.cs_next.as_ref().is_some_and(|(k, _)| *k == min_key) {
                self.cs_next = self.cursor.next()?;
            }
            if self.hist_next_key.as_ref().is_some_and(|k| *k == min_key) {
                self.hist_next_key = self.advance_history_past(&min_key)?;
            }

            // Resolve the value at max_block_number.
            if let Some(account) = self.resolve_account_merge(min_key, cs_value.as_ref())? {
                return Ok(Some((min_key, account)));
            }
            // Key doesn't exist at max_block_number — continue to next.
        }
    }
}

impl<C, HC, CC> HashedCursor for V2AccountCursor<C, HC, CC>
where
    C: DbCursorRO<V2HashedAccounts> + Send,
    HC: DbCursorRO<V2HashedAccountsHistory> + Send,
    CC: DbCursorRO<V2HashedAccountChangeSets> + DbDupCursorRO<V2HashedAccountChangeSets> + Send,
{
    type Value = Account;

    fn seek(&mut self, key: B256) -> Result<Option<(B256, Self::Value)>, DatabaseError> {
        self.seeked = true;

        if self.is_latest {
            // Fast path: current state is authoritative, no history needed.
            return self.cursor.seek(key);
        }

        // Initialize both merge cursors at the target key.
        self.cs_next = self.cursor.seek(key)?;
        self.hist_next_key = self
            .history_walk_cursor
            .seek(HashedAccountShardedKey::new(key, 0))?
            .map(|(k, _)| k.0.key);
        self.find_next_live()
    }

    fn next(&mut self) -> Result<Option<(B256, Self::Value)>, DatabaseError> {
        if !self.seeked {
            return self.seek(B256::ZERO);
        }

        if self.is_latest {
            return self.cursor.next();
        }

        self.find_next_live()
    }

    fn reset(&mut self) {
        self.cs_next = None;
        self.hist_next_key = None;
        self.seeked = false;
    }
}

/// History-aware cursor over the [`V2HashedStorages`] v2 `DupSort` table.
///
/// Uses the same dual-cursor merge strategy as [`V2AccountCursor`] but
/// scoped to a single `hashed_address`. Both the current-state `DupSort`
/// entries and the history-bitmap entries are walked in parallel to discover
/// storage slots that may have been deleted after `max_block_number`.
#[derive(Debug)]
pub struct V2StorageCursor<C, HC, CC> {
    /// Current state cursor (`DupSort`).
    cursor: C,
    /// History bitmap cursor for resolving individual keys.
    history_cursor: HC,
    /// History bitmap cursor for merge-walking deleted keys.
    history_walk_cursor: HC,
    /// Changeset cursor (`DupSort`).
    changeset_cursor: CC,
    /// Target hashed address.
    hashed_address: B256,
    /// Target block number for historical reads.
    max_block_number: u64,
    /// Pre-fetched next entry from the current state walk (within address).
    cs_next: Option<reth_primitives_traits::StorageEntry>,
    /// Pre-fetched next unique storage key from the history walk.
    hist_next_key: Option<B256>,
    /// Whether `seek` has been called to initialize the merge cursors.
    seeked: bool,
    /// Fast path: when `true`, skip all history/changeset lookups.
    is_latest: bool,
}

impl<C, HC, CC> V2StorageCursor<C, HC, CC> {
    /// Create a new [`V2StorageCursor`].
    pub const fn new(
        cursor: C,
        history_cursor: HC,
        history_walk_cursor: HC,
        changeset_cursor: CC,
        hashed_address: B256,
        max_block_number: u64,
        is_latest: bool,
    ) -> Self {
        Self {
            cursor,
            history_cursor,
            history_walk_cursor,
            changeset_cursor,
            hashed_address,
            max_block_number,
            cs_next: None,
            hist_next_key: None,
            seeked: false,
            is_latest,
        }
    }
}

impl<C, HC, CC> V2StorageCursor<C, HC, CC>
where
    C: DbCursorRO<V2HashedStorages> + DbDupCursorRO<V2HashedStorages>,
    HC: DbCursorRO<V2HashedStoragesHistory>,
    CC: DbCursorRO<V2HashedStorageChangeSets> + DbDupCursorRO<V2HashedStorageChangeSets>,
{
    /// Resolve a storage slot using a pre-fetched current-state value.
    ///
    /// Does **not** touch the walk cursor, so it is safe to call from the
    /// merge loop (`find_next_live`).
    fn resolve_storage_merge(
        &mut self,
        storage_key: B256,
        cs_value: Option<&U256>,
    ) -> Result<Option<U256>, DatabaseError> {
        let history_key = HashedStorageShardedKey {
            hashed_address: self.hashed_address,
            sharded_key: ShardedKey::new(storage_key, self.max_block_number),
        };

        let addr = self.hashed_address;
        let source = find_source::<V2HashedStoragesHistory, _>(
            &mut self.history_cursor,
            history_key,
            self.max_block_number,
            |k| k.hashed_address == addr && k.sharded_key.key == storage_key,
        )?;

        match source {
            ResolvedSource::FromChangeset(changeset_block) => {
                let cs_key = BlockNumberHashedAddress((changeset_block, self.hashed_address));
                let entry = self
                    .changeset_cursor
                    .seek_by_key_subkey(cs_key, storage_key)?
                    .filter(|e| e.key == storage_key);
                match entry {
                    Some(e) if e.value.is_zero() => Ok(None),
                    Some(e) => Ok(Some(e.value)),
                    None => Ok(None),
                }
            }
            ResolvedSource::FromCurrentState => {
                Ok(cs_value.copied().filter(|v| !v.is_zero()))
            }
        }
    }

    /// Advance the history walk cursor past all shards of `key` (for this
    /// address) and return the next distinct storage key, if any.
    fn advance_history_past(
        &mut self,
        key: &B256,
    ) -> Result<Option<B256>, DatabaseError> {
        let seek = HashedStorageShardedKey {
            hashed_address: self.hashed_address,
            sharded_key: ShardedKey::new(*key, u64::MAX),
        };
        let entry = self
            .history_walk_cursor
            .seek(seek)?
            .filter(|(k, _)| k.hashed_address == self.hashed_address);
        match entry {
            Some((k, _)) if k.sharded_key.key == *key => {
                // On the last shard of this key — advance once more.
                Ok(self
                    .history_walk_cursor
                    .next()?
                    .filter(|(k, _)| k.hashed_address == self.hashed_address)
                    .map(|(k, _)| k.sharded_key.key))
            }
            Some((k, _)) => Ok(Some(k.sharded_key.key)),
            None => Ok(None),
        }
    }

    /// Merge-walk both the current-state `DupSort` cursor and the history-bitmap
    /// cursor, yielding the next storage slot whose value is live at
    /// `max_block_number`.
    fn find_next_live(
        &mut self,
    ) -> Result<Option<(B256, U256)>, DatabaseError> {
        loop {
            let (min_key, cs_value) = match (&self.cs_next, &self.hist_next_key) {
                (Some(cs_entry), Some(h_k)) => {
                    if cs_entry.key <= *h_k {
                        (cs_entry.key, Some(cs_entry.value))
                    } else {
                        (*h_k, None)
                    }
                }
                (Some(cs_entry), None) => (cs_entry.key, Some(cs_entry.value)),
                (None, Some(h_k)) => (*h_k, None),
                (None, None) => return Ok(None),
            };

            // Advance whichever cursor(s) produced this key.
            if self.cs_next.as_ref().is_some_and(|e| e.key == min_key) {
                self.cs_next = self.cursor.next_dup_val()?;
            }
            if self.hist_next_key.as_ref().is_some_and(|k| *k == min_key) {
                self.hist_next_key = self.advance_history_past(&min_key)?;
            }

            // Resolve the value at max_block_number.
            if let Some(value) = self.resolve_storage_merge(min_key, cs_value.as_ref())? {
                return Ok(Some((min_key, value)));
            }
            // Key doesn't exist at max_block_number — continue to next.
        }
    }
}

impl<C, HC, CC> HashedCursor for V2StorageCursor<C, HC, CC>
where
    C: DbCursorRO<V2HashedStorages> + DbDupCursorRO<V2HashedStorages> + Send,
    HC: DbCursorRO<V2HashedStoragesHistory> + Send,
    CC: DbCursorRO<V2HashedStorageChangeSets>
        + DbDupCursorRO<V2HashedStorageChangeSets>
        + Send,
{
    type Value = U256;

    fn seek(&mut self, subkey: B256) -> Result<Option<(B256, Self::Value)>, DatabaseError> {
        self.seeked = true;

        if self.is_latest {
            // Fast path: current state is authoritative.
            // Loop to skip zero-valued entries (tombstones).
            let mut entry =
                self.cursor.seek_by_key_subkey(self.hashed_address, subkey)?;
            while let Some(ref e) = entry {
                if !e.value.is_zero() {
                    return Ok(Some((e.key, e.value)));
                }
                entry = self.cursor.next_dup_val()?;
            }
            return Ok(None);
        }

        // Initialize both merge cursors at the target key.
        self.cs_next = self.cursor.seek_by_key_subkey(self.hashed_address, subkey)?;
        let hist_seek = HashedStorageShardedKey {
            hashed_address: self.hashed_address,
            sharded_key: ShardedKey::new(subkey, 0),
        };
        self.hist_next_key = self
            .history_walk_cursor
            .seek(hist_seek)?
            .filter(|(k, _)| k.hashed_address == self.hashed_address)
            .map(|(k, _)| k.sharded_key.key);
        self.find_next_live()
    }

    fn next(&mut self) -> Result<Option<(B256, Self::Value)>, DatabaseError> {
        if !self.seeked {
            return self.seek(B256::ZERO);
        }

        if self.is_latest {
            // Loop to skip zero-valued entries (tombstones).
            while let Some(e) = self.cursor.next_dup_val()? {
                if !e.value.is_zero() {
                    return Ok(Some((e.key, e.value)));
                }
            }
            return Ok(None);
        }

        self.find_next_live()
    }

    fn reset(&mut self) {
        self.cs_next = None;
        self.hist_next_key = None;
        self.seeked = false;
    }
}

impl<C, HC, CC> HashedStorageCursor for V2StorageCursor<C, HC, CC>
where
    C: DbCursorRO<V2HashedStorages> + DbDupCursorRO<V2HashedStorages> + Send,
    HC: DbCursorRO<V2HashedStoragesHistory> + Send,
    CC: DbCursorRO<V2HashedStorageChangeSets>
        + DbDupCursorRO<V2HashedStorageChangeSets>
        + Send,
{
    fn is_storage_empty(&mut self) -> Result<bool, DatabaseError> {
        Ok(self.seek(B256::ZERO)?.is_none())
    }

    fn set_hashed_address(&mut self, hashed_address: B256) {
        self.hashed_address = hashed_address;
        self.cs_next = None;
        self.hist_next_key = None;
        self.seeked = false;
    }
}

/// History-aware cursor over the [`V2AccountsTrie`] v2 tables.
///
/// Uses a **dual-cursor merge** to discover all trie paths that existed at
/// `max_block_number`. This is necessary because a key deleted *after* the
/// target block no longer exists in the current-state table and would be
/// missed by a walk of current state alone. The merge walks both the
/// current-state cursor and the history-bitmap cursor in sorted order,
/// yielding the minimum key from each, resolving its value at the target
/// block, and skipping keys that did not exist at that block.
#[derive(Debug)]
pub struct V2AccountTrieCursor<C, HC, CC> {
    /// Current state walk cursor.
    cursor: C,
    /// History bitmap cursor for resolving individual keys.
    history_cursor: HC,
    /// History bitmap cursor for merge-walking deleted keys.
    history_walk_cursor: HC,
    /// Changeset cursor.
    changeset_cursor: CC,
    /// Target block number.
    max_block_number: u64,
    /// Pre-fetched next entry from the current state walk.
    cs_next: Option<(StoredNibbles, BranchNodeCompact)>,
    /// Pre-fetched next unique key from the history walk.
    hist_next_key: Option<StoredNibbles>,
    /// Last key yielded by `seek`/`next` (for `current()`).
    last_key: Option<StoredNibbles>,
    /// Whether `seek` or `seek_exact` has been called to initialize the merge cursors.
    seeked: bool,
    /// Fast path: when `true`, skip all history/changeset lookups.
    is_latest: bool,
}

impl<C, HC, CC> V2AccountTrieCursor<C, HC, CC> {
    /// Create a new [`V2AccountTrieCursor`].
    pub const fn new(
        cursor: C,
        history_cursor: HC,
        history_walk_cursor: HC,
        changeset_cursor: CC,
        max_block_number: u64,
        is_latest: bool,
    ) -> Self {
        Self {
            cursor,
            history_cursor,
            history_walk_cursor,
            changeset_cursor,
            max_block_number,
            cs_next: None,
            hist_next_key: None,
            last_key: None,
            seeked: false,
            is_latest,
        }
    }
}

impl<C, HC, CC> V2AccountTrieCursor<C, HC, CC>
where
    C: DbCursorRO<V2AccountsTrie>,
    HC: DbCursorRO<V2AccountsTrieHistory>,
    CC: DbCursorRO<V2AccountTrieChangeSets> + DbDupCursorRO<V2AccountTrieChangeSets>,
{
    /// Resolve a key using the walk cursor for the `FromCurrentState` case.
    ///
    /// May disrupt the walk cursor position — only call when the walk state
    /// will be re-synced immediately afterward (e.g. in `seek_exact`).
    fn resolve_node_standalone(
        &mut self,
        path: &StoredNibbles,
    ) -> Result<Option<BranchNodeCompact>, DatabaseError> {
        let seek_key = AccountTrieShardedKey::new(path.clone(), self.max_block_number);
        let target = path.clone();
        let source = find_source::<V2AccountsTrieHistory, _>(
            &mut self.history_cursor,
            seek_key,
            self.max_block_number,
            |k| k.key == target,
        )?;

        match source {
            ResolvedSource::FromChangeset(changeset_block) => {
                let entry = self
                    .changeset_cursor
                    .seek_by_key_subkey(changeset_block, StoredNibblesSubKey(path.0))?
                    .filter(|e| e.nibbles == StoredNibblesSubKey(path.0));
                Ok(entry.and_then(|e| e.node))
            }
            ResolvedSource::FromCurrentState => {
                Ok(self.cursor.seek_exact(path.clone())?.map(|(_, node)| node))
            }
        }
    }

    /// Resolve a key using a pre-fetched current-state value.
    ///
    /// Does **not** touch the walk cursor, so it is safe to call from the
    /// merge loop (`find_next_live`).
    fn resolve_node_merge(
        &mut self,
        path: &StoredNibbles,
        cs_value: Option<&BranchNodeCompact>,
    ) -> Result<Option<BranchNodeCompact>, DatabaseError> {
        let seek_key = AccountTrieShardedKey::new(path.clone(), self.max_block_number);
        let target = path.clone();
        let source = find_source::<V2AccountsTrieHistory, _>(
            &mut self.history_cursor,
            seek_key,
            self.max_block_number,
            |k| k.key == target,
        )?;

        match source {
            ResolvedSource::FromChangeset(changeset_block) => {
                let entry = self
                    .changeset_cursor
                    .seek_by_key_subkey(changeset_block, StoredNibblesSubKey(path.0))?
                    .filter(|e| e.nibbles == StoredNibblesSubKey(path.0));
                Ok(entry.and_then(|e| e.node))
            }
            ResolvedSource::FromCurrentState => Ok(cs_value.cloned()),
        }
    }

    /// Advance the history walk cursor past all shards of `key` and return
    /// the next distinct key, if any.
    fn advance_history_past(
        &mut self,
        key: &StoredNibbles,
    ) -> Result<Option<StoredNibbles>, DatabaseError> {
        // Jump to the last shard of this key (or past it entirely).
        let entry =
            self.history_walk_cursor.seek(AccountTrieShardedKey::new(key.clone(), u64::MAX))?;
        match entry {
            Some((k, _)) if k.key == *key => {
                // On the last shard of this key — one more step to reach the
                // next distinct key.
                Ok(self.history_walk_cursor.next()?.map(|(k, _)| k.key))
            }
            Some((k, _)) => Ok(Some(k.key)),
            None => Ok(None),
        }
    }

    /// Merge-walk both the current-state cursor and the history-bitmap cursor,
    /// yielding the next key (in ascending order) whose value is live at
    /// `max_block_number`.
    fn find_next_live(
        &mut self,
    ) -> Result<Option<(Nibbles, BranchNodeCompact)>, DatabaseError> {
        loop {
            // Pick the minimum key from the two sources.
            let (min_key, cs_value) = match (&self.cs_next, &self.hist_next_key) {
                (Some((cs_k, cs_v)), Some(h_k)) => {
                    if cs_k <= h_k {
                        (cs_k.clone(), Some(cs_v.clone()))
                    } else {
                        (h_k.clone(), None)
                    }
                }
                (Some((cs_k, cs_v)), None) => (cs_k.clone(), Some(cs_v.clone())),
                (None, Some(h_k)) => (h_k.clone(), None),
                (None, None) => return Ok(None),
            };

            // Advance whichever cursor(s) produced this key.
            if self.cs_next.as_ref().is_some_and(|(k, _)| *k == min_key) {
                self.cs_next = self.cursor.next()?;
            }
            if self.hist_next_key.as_ref().is_some_and(|k| *k == min_key) {
                self.hist_next_key = self.advance_history_past(&min_key)?;
            }

            // Resolve the value at max_block_number.
            if let Some(node) = self.resolve_node_merge(&min_key, cs_value.as_ref())? {
                self.last_key = Some(min_key.clone());
                return Ok(Some((min_key.0, node)));
            }
            // Key doesn't exist at max_block_number — continue to next.
        }
    }
}

impl<C, HC, CC> TrieCursor for V2AccountTrieCursor<C, HC, CC>
where
    C: DbCursorRO<V2AccountsTrie> + Send,
    HC: DbCursorRO<V2AccountsTrieHistory> + Send,
    CC: DbCursorRO<V2AccountTrieChangeSets>
        + DbDupCursorRO<V2AccountTrieChangeSets>
        + Send,
{
    fn seek_exact(
        &mut self,
        key: Nibbles,
    ) -> Result<Option<(Nibbles, BranchNodeCompact)>, DatabaseError> {
        self.seeked = true;

        if self.is_latest {
            // Fast path: direct current-state lookup.
            let result = self.cursor.seek_exact(StoredNibbles(key))?;
            if result.is_some() {
                self.last_key = Some(StoredNibbles(key));
            }
            return Ok(result.map(|(_, node)| (key, node)));
        }

        let path = StoredNibbles(key);
        let node = self.resolve_node_standalone(&path)?;

        // Re-sync the walk state so a subsequent next() starts after `path`.
        let cs_at_key = self.cursor.seek(path.clone())?;
        self.cs_next = match cs_at_key {
            Some((k, _)) if k == path => self.cursor.next()?,
            other => other,
        };
        self.hist_next_key = self.advance_history_past(&path)?;

        if node.is_some() {
            self.last_key = Some(path);
        }
        Ok(node.map(|n| (key, n)))
    }

    fn seek(
        &mut self,
        key: Nibbles,
    ) -> Result<Option<(Nibbles, BranchNodeCompact)>, DatabaseError> {
        self.seeked = true;

        if self.is_latest {
            // Fast path: direct current-state walk.
            let result = self.cursor.seek(StoredNibbles(key))?;
            if let Some((ref k, _)) = result {
                self.last_key = Some(k.clone());
            }
            return Ok(result.map(|(k, node)| (k.0, node)));
        }

        // Initialize both merge cursors at the target key.
        self.cs_next = self.cursor.seek(StoredNibbles(key))?;
        self.hist_next_key = self
            .history_walk_cursor
            .seek(AccountTrieShardedKey::new(StoredNibbles(key), 0))?
            .map(|(k, _)| k.key);
        self.find_next_live()
    }

    fn next(&mut self) -> Result<Option<(Nibbles, BranchNodeCompact)>, DatabaseError> {
        if !self.seeked {
            return self.seek(Nibbles::default());
        }

        if self.is_latest {
            let result = self.cursor.next()?;
            if let Some((ref k, _)) = result {
                self.last_key = Some(k.clone());
            }
            return Ok(result.map(|(k, node)| (k.0, node)));
        }

        self.find_next_live()
    }

    fn current(&mut self) -> Result<Option<Nibbles>, DatabaseError> {
        Ok(self.last_key.as_ref().map(|k| k.0))
    }

    fn reset(&mut self) {}
}

/// History-aware cursor over the [`V2StoragesTrie`] v2 `DupSort` table.
///
/// Uses the same dual-cursor merge strategy as [`V2AccountTrieCursor`] but
/// scoped to a single `hashed_address`. Both the current-state `DupSort`
/// entries and the history-bitmap entries are walked in parallel to discover
/// keys that may have been deleted after `max_block_number`.
#[derive(Debug)]
pub struct V2StorageTrieCursor<C, HC, CC> {
    /// Current state cursor (`DupSort`).
    cursor: C,
    /// History bitmap cursor for resolving individual keys.
    history_cursor: HC,
    /// History bitmap cursor for merge-walking deleted keys.
    history_walk_cursor: HC,
    /// Changeset cursor (`DupSort`).
    changeset_cursor: CC,
    /// Target hashed address.
    hashed_address: B256,
    /// Target block number.
    max_block_number: u64,
    /// Pre-fetched next entry from the current state walk (within address).
    cs_next: Option<StorageTrieEntry>,
    /// Pre-fetched next unique nibbles key from the history walk.
    hist_next_key: Option<StoredNibbles>,
    /// Last key yielded by `seek`/`next` (for `current()`).
    last_key: Option<StoredNibbles>,
    /// Whether `seek` or `seek_exact` has been called to initialize the merge cursors.
    seeked: bool,
    /// Fast path: when `true`, skip all history/changeset lookups.
    is_latest: bool,
}

impl<C, HC, CC> V2StorageTrieCursor<C, HC, CC> {
    /// Create a new [`V2StorageTrieCursor`].
    pub const fn new(
        cursor: C,
        history_cursor: HC,
        history_walk_cursor: HC,
        changeset_cursor: CC,
        hashed_address: B256,
        max_block_number: u64,
        is_latest: bool,
    ) -> Self {
        Self {
            cursor,
            history_cursor,
            history_walk_cursor,
            changeset_cursor,
            hashed_address,
            max_block_number,
            cs_next: None,
            hist_next_key: None,
            last_key: None,
            seeked: false,
            is_latest,
        }
    }
}

impl<C, HC, CC> V2StorageTrieCursor<C, HC, CC>
where
    C: DbCursorRO<V2StoragesTrie> + DbDupCursorRO<V2StoragesTrie>,
    HC: DbCursorRO<V2StoragesTrieHistory>,
    CC: DbCursorRO<V2StorageTrieChangeSets> + DbDupCursorRO<V2StorageTrieChangeSets>,
{
    /// Resolve a key using the walk cursor for the `FromCurrentState` case.
    ///
    /// May disrupt the walk cursor position — only call from `seek_exact`.
    fn resolve_node_standalone(
        &mut self,
        path: Nibbles,
    ) -> Result<Option<BranchNodeCompact>, DatabaseError> {
        let nibbles = StoredNibbles(path);
        let seek_key = StorageTrieShardedKey::new(
            self.hashed_address,
            nibbles.clone(),
            self.max_block_number,
        );

        let addr = self.hashed_address;
        let nibbles_cmp = nibbles;
        let source = find_source::<V2StoragesTrieHistory, _>(
            &mut self.history_cursor,
            seek_key,
            self.max_block_number,
            |k| k.hashed_address == addr && k.key == nibbles_cmp,
        )?;

        match source {
            ResolvedSource::FromChangeset(changeset_block) => {
                let cs_key = BlockNumberHashedAddress((changeset_block, self.hashed_address));
                let entry = self
                    .changeset_cursor
                    .seek_by_key_subkey(cs_key, StoredNibblesSubKey(path))?
                    .filter(|e| e.nibbles == StoredNibblesSubKey(path));
                Ok(entry.and_then(|e| e.node))
            }
            ResolvedSource::FromCurrentState => Ok(self
                .cursor
                .seek_by_key_subkey(self.hashed_address, StoredNibblesSubKey(path))?
                .filter(|e| e.nibbles == StoredNibblesSubKey(path))
                .map(|e| e.node)),
        }
    }

    /// Resolve a key using a pre-fetched current-state value.
    ///
    /// Does **not** touch the walk cursor.
    fn resolve_node_merge(
        &mut self,
        path: Nibbles,
        cs_value: Option<&BranchNodeCompact>,
    ) -> Result<Option<BranchNodeCompact>, DatabaseError> {
        let nibbles = StoredNibbles(path);
        let seek_key = StorageTrieShardedKey::new(
            self.hashed_address,
            nibbles.clone(),
            self.max_block_number,
        );

        let addr = self.hashed_address;
        let nibbles_cmp = nibbles;
        let source = find_source::<V2StoragesTrieHistory, _>(
            &mut self.history_cursor,
            seek_key,
            self.max_block_number,
            |k| k.hashed_address == addr && k.key == nibbles_cmp,
        )?;

        match source {
            ResolvedSource::FromChangeset(changeset_block) => {
                let cs_key = BlockNumberHashedAddress((changeset_block, self.hashed_address));
                let entry = self
                    .changeset_cursor
                    .seek_by_key_subkey(cs_key, StoredNibblesSubKey(path))?
                    .filter(|e| e.nibbles == StoredNibblesSubKey(path));
                Ok(entry.and_then(|e| e.node))
            }
            ResolvedSource::FromCurrentState => Ok(cs_value.cloned()),
        }
    }

    /// Advance the history walk cursor past all shards of `key` (for this
    /// address) and return the next distinct nibbles key, if any.
    fn advance_history_past(
        &mut self,
        key: &StoredNibbles,
    ) -> Result<Option<StoredNibbles>, DatabaseError> {
        let seek = StorageTrieShardedKey::new(
            self.hashed_address,
            key.clone(),
            u64::MAX,
        );
        let entry = self
            .history_walk_cursor
            .seek(seek)?
            .filter(|(k, _)| k.hashed_address == self.hashed_address);
        match entry {
            Some((k, _)) if k.key == *key => {
                // On the last shard of this key — advance once more.
                Ok(self
                    .history_walk_cursor
                    .next()?
                    .filter(|(k, _)| k.hashed_address == self.hashed_address)
                    .map(|(k, _)| k.key))
            }
            Some((k, _)) => Ok(Some(k.key)),
            None => Ok(None),
        }
    }

    /// Merge-walk both the current-state `DupSort` cursor and the history-bitmap
    /// cursor, yielding the next path whose node is live at `max_block_number`.
    fn find_next_live(
        &mut self,
    ) -> Result<Option<(Nibbles, BranchNodeCompact)>, DatabaseError> {
        loop {
            let (min_nibbles, cs_node) = match (&self.cs_next, &self.hist_next_key) {
                (Some(cs_entry), Some(h_k)) => {
                    let cs_stored = StoredNibbles(cs_entry.nibbles.0);
                    if cs_stored <= *h_k {
                        (cs_stored, Some(cs_entry.node.clone()))
                    } else {
                        (h_k.clone(), None)
                    }
                }
                (Some(cs_entry), None) => {
                    (StoredNibbles(cs_entry.nibbles.0), Some(cs_entry.node.clone()))
                }
                (None, Some(h_k)) => (h_k.clone(), None),
                (None, None) => return Ok(None),
            };

            // Advance whichever cursor(s) produced this key.
            if self
                .cs_next
                .as_ref()
                .is_some_and(|e| StoredNibbles(e.nibbles.0) == min_nibbles)
            {
                self.cs_next = self.cursor.next_dup()?.map(|(_, v)| v);
            }
            if self.hist_next_key.as_ref().is_some_and(|k| *k == min_nibbles) {
                self.hist_next_key = self.advance_history_past(&min_nibbles)?;
            }

            // Resolve the value at max_block_number.
            if let Some(node) =
                self.resolve_node_merge(min_nibbles.0, cs_node.as_ref())?
            {
                self.last_key = Some(StoredNibbles(min_nibbles.0));
                return Ok(Some((min_nibbles.0, node)));
            }
        }
    }
}

impl<C, HC, CC> TrieCursor for V2StorageTrieCursor<C, HC, CC>
where
    C: DbCursorRO<V2StoragesTrie> + DbDupCursorRO<V2StoragesTrie> + Send,
    HC: DbCursorRO<V2StoragesTrieHistory> + Send,
    CC: DbCursorRO<V2StorageTrieChangeSets>
        + DbDupCursorRO<V2StorageTrieChangeSets>
        + Send,
{
    fn seek_exact(
        &mut self,
        key: Nibbles,
    ) -> Result<Option<(Nibbles, BranchNodeCompact)>, DatabaseError> {
        self.seeked = true;

        if self.is_latest {
            // Fast path: direct DupSort lookup.
            let entry = self
                .cursor
                .seek_by_key_subkey(self.hashed_address, StoredNibblesSubKey(key))?
                .filter(|e| e.nibbles == StoredNibblesSubKey(key));
            if entry.is_some() {
                self.last_key = Some(StoredNibbles(key));
            }
            return Ok(entry.map(|e| (key, e.node)));
        }

        let node = self.resolve_node_standalone(key)?;

        // Re-sync walk state so a subsequent next() starts after `key`.
        let cs_at_key =
            self.cursor.seek_by_key_subkey(self.hashed_address, StoredNibblesSubKey(key))?;
        self.cs_next = match cs_at_key {
            Some(e) if e.nibbles == StoredNibblesSubKey(key) => {
                self.cursor.next_dup()?.map(|(_, v)| v)
            }
            other => other,
        };
        let path = StoredNibbles(key);
        self.hist_next_key = self.advance_history_past(&path)?;

        if node.is_some() {
            self.last_key = Some(path);
        }
        Ok(node.map(|n| (key, n)))
    }

    fn seek(
        &mut self,
        key: Nibbles,
    ) -> Result<Option<(Nibbles, BranchNodeCompact)>, DatabaseError> {
        self.seeked = true;

        if self.is_latest {
            // Fast path: direct DupSort walk.
            let entry =
                self.cursor.seek_by_key_subkey(self.hashed_address, StoredNibblesSubKey(key))?;
            if let Some(ref e) = entry {
                self.last_key = Some(StoredNibbles(e.nibbles.0));
            }
            return Ok(entry.map(|e| (e.nibbles.0, e.node)));
        }

        // Initialize both merge cursors at the target key.
        self.cs_next =
            self.cursor.seek_by_key_subkey(self.hashed_address, StoredNibblesSubKey(key))?;
        let hist_seek = StorageTrieShardedKey::new(
            self.hashed_address,
            StoredNibbles(key),
            0,
        );
        self.hist_next_key = self
            .history_walk_cursor
            .seek(hist_seek)?
            .filter(|(k, _)| k.hashed_address == self.hashed_address)
            .map(|(k, _)| k.key);
        self.find_next_live()
    }

    fn next(&mut self) -> Result<Option<(Nibbles, BranchNodeCompact)>, DatabaseError> {
        if !self.seeked {
            return self.seek(Nibbles::default());
        }

        if self.is_latest {
            let entry = self.cursor.next_dup()?.map(|(_, v)| v);
            if let Some(ref e) = entry {
                self.last_key = Some(StoredNibbles(e.nibbles.0));
            }
            return Ok(entry.map(|e| (e.nibbles.0, e.node)));
        }

        self.find_next_live()
    }

    fn current(&mut self) -> Result<Option<Nibbles>, DatabaseError> {
        Ok(self.last_key.as_ref().map(|k| k.0))
    }

    fn reset(&mut self) {}
}

impl<C, HC, CC> TrieStorageCursor for V2StorageTrieCursor<C, HC, CC>
where
    C: DbCursorRO<V2StoragesTrie> + DbDupCursorRO<V2StoragesTrie> + Send,
    HC: DbCursorRO<V2StoragesTrieHistory> + Send,
    CC: DbCursorRO<V2StorageTrieChangeSets>
        + DbDupCursorRO<V2StorageTrieChangeSets>
        + Send,
{
    fn set_hashed_address(&mut self, hashed_address: B256) {
        self.hashed_address = hashed_address;
        self.cs_next = None;
        self.hist_next_key = None;
        self.last_key = None;
        self.seeked = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::models;
    use crate::db::models::{HashedAccountBeforeTx, TrieChangeSetsEntry};
    use reth_db::{
        cursor::{DbCursorRW, DbDupCursorRW},
        mdbx::{init_db_for, DatabaseArguments},
        Database, DatabaseEnv,
    };
    use reth_db_api::transaction::{DbTx, DbTxMut};
    use reth_primitives_traits::StorageEntry;
    use reth_trie::{
        BranchNodeCompact, Nibbles, StoredNibbles, StoredNibblesSubKey,
    };
    use tempfile::TempDir;

    fn setup_db() -> DatabaseEnv {
        let tmp = TempDir::new().expect("create tmpdir");
        init_db_for::<_, models::Tables>(tmp, DatabaseArguments::default()).expect("init db")
    }

    fn node() -> BranchNodeCompact {
        BranchNodeCompact::new(0b11, 0, 0, vec![], Some(B256::repeat_byte(0xAB)))
    }

    fn node2() -> BranchNodeCompact {
        BranchNodeCompact::new(0b101, 0, 0, vec![], Some(B256::repeat_byte(0xCD)))
    }

    fn sample_account(nonce: u64) -> Account {
        Account { nonce, ..Default::default() }
    }

    // ====================== find_source unit tests ======================

    #[test]
    fn find_source_returns_current_state_when_no_history() {
        let db = setup_db();
        let addr = B256::from([0xAA; 32]);

        let tx = db.tx().expect("ro tx");
        let mut cursor = tx.cursor_read::<V2HashedAccountsHistory>().expect("c");

        let result = find_source::<V2HashedAccountsHistory, _>(
            &mut cursor,
            HashedAccountShardedKey::new(addr, 10),
            10,
            |k| k.0.key == addr,
        )
        .expect("ok");

        assert_eq!(result, ResolvedSource::FromCurrentState);
    }

    #[test]
    fn find_source_returns_changeset_when_modification_after_target() {
        let db = setup_db();
        let addr = B256::from([0xBB; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            wtx.cursor_write::<V2HashedAccountsHistory>()
                .expect("c")
                .upsert(
                    HashedAccountShardedKey::new(addr, u64::MAX),
                    &BlockNumberList::new_pre_sorted([5, 10, 15]),
                )
                .expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cursor = tx.cursor_read::<V2HashedAccountsHistory>().expect("c");

        // Target block 7 → first block > 7 in [5, 10, 15] is 10
        let result = find_source::<V2HashedAccountsHistory, _>(
            &mut cursor,
            HashedAccountShardedKey::new(addr, 7),
            7,
            |k| k.0.key == addr,
        )
        .expect("ok");

        assert_eq!(result, ResolvedSource::FromChangeset(10));
    }

    #[test]
    fn find_source_returns_current_state_when_no_modification_after_target() {
        let db = setup_db();
        let addr = B256::from([0xCC; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            wtx.cursor_write::<V2HashedAccountsHistory>()
                .expect("c")
                .upsert(
                    HashedAccountShardedKey::new(addr, u64::MAX),
                    &BlockNumberList::new_pre_sorted([3, 7]),
                )
                .expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cursor = tx.cursor_read::<V2HashedAccountsHistory>().expect("c");

        // Target block 10 → no block > 10 in [3, 7]
        let result = find_source::<V2HashedAccountsHistory, _>(
            &mut cursor,
            HashedAccountShardedKey::new(addr, 10),
            10,
            |k| k.0.key == addr,
        )
        .expect("ok");

        assert_eq!(result, ResolvedSource::FromCurrentState);
    }

    #[test]
    fn find_source_handles_exact_match_block() {
        let db = setup_db();
        let addr = B256::from([0xDD; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            wtx.cursor_write::<V2HashedAccountsHistory>()
                .expect("c")
                .upsert(
                    HashedAccountShardedKey::new(addr, u64::MAX),
                    &BlockNumberList::new_pre_sorted([5, 10, 15]),
                )
                .expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cursor = tx.cursor_read::<V2HashedAccountsHistory>().expect("c");

        // Target block 10 (exactly in the bitmap) → first block > 10 is 15
        let result = find_source::<V2HashedAccountsHistory, _>(
            &mut cursor,
            HashedAccountShardedKey::new(addr, 10),
            10,
            |k| k.0.key == addr,
        )
        .expect("ok");

        assert_eq!(result, ResolvedSource::FromChangeset(15));
    }

    // ====================== find_source with AccountTrieShardedKey tests ======================

    #[test]
    fn find_source_resolves_root_path_despite_child_history() {
        // Regression test: the root trie path [] has history at blocks [10, 15].
        // A child path [0] also has history. With the old `ShardedKey<StoredNibbles>`
        // encoding (no length prefix), `cursor.seek` would land on the wrong path.
        // With `AccountTrieShardedKey`'s length-prefixed encoding, `find_source` works
        // correctly: all shards of [] sort before all shards of [0].
        let db = setup_db();
        let root_path = StoredNibbles(Nibbles::default());
        let child_path = StoredNibbles(Nibbles::from_nibbles([0]));

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut cursor = wtx.cursor_write::<V2AccountsTrieHistory>().expect("c");
            // Root path history: modified at blocks 10, 15
            cursor
                .upsert(
                    AccountTrieShardedKey::new(root_path.clone(), u64::MAX),
                    &BlockNumberList::new_pre_sorted([10, 15]),
                )
                .expect("upsert root");
            // Child path [0] history: modified at blocks 10, 15
            cursor
                .upsert(
                    AccountTrieShardedKey::new(child_path, u64::MAX),
                    &BlockNumberList::new_pre_sorted([10, 15]),
                )
                .expect("upsert child");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cursor = tx.cursor_read::<V2AccountsTrieHistory>().expect("c");

        // Query at block 12 — should find changeset at block 15
        // (the first modification after block 12).
        let result = find_source::<V2AccountsTrieHistory, _>(
            &mut cursor,
            AccountTrieShardedKey::new(root_path.clone(), 12),
            12,
            |k| k.key == root_path,
        )
        .expect("ok");

        assert_eq!(result, ResolvedSource::FromChangeset(15));
    }

    #[test]
    fn find_source_trie_returns_current_state_when_no_history() {
        let db = setup_db();
        let root_path = StoredNibbles(Nibbles::default());

        let tx = db.tx().expect("ro tx");
        let mut cursor = tx.cursor_read::<V2AccountsTrieHistory>().expect("c");

        let result = find_source::<V2AccountsTrieHistory, _>(
            &mut cursor,
            AccountTrieShardedKey::new(root_path.clone(), 10),
            10,
            |k| k.key == root_path,
        )
        .expect("ok");

        assert_eq!(result, ResolvedSource::FromCurrentState);
    }

    #[test]
    fn find_source_trie_returns_current_state_when_all_modifications_before_target() {
        let db = setup_db();
        let root_path = StoredNibbles(Nibbles::default());

        {
            let wtx = db.tx_mut().expect("rw tx");
            wtx.cursor_write::<V2AccountsTrieHistory>()
                .expect("c")
                .upsert(
                    AccountTrieShardedKey::new(root_path.clone(), u64::MAX),
                    &BlockNumberList::new_pre_sorted([5, 8]),
                )
                .expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cursor = tx.cursor_read::<V2AccountsTrieHistory>().expect("c");

        // Target block 10 — all modifications (5, 8) are ≤ 10
        let result = find_source::<V2AccountsTrieHistory, _>(
            &mut cursor,
            AccountTrieShardedKey::new(root_path.clone(), 10),
            10,
            |k| k.key == root_path,
        )
        .expect("ok");

        assert_eq!(result, ResolvedSource::FromCurrentState);
    }

    #[test]
    fn find_source_handles_root_path_with_child_history() {
        // Verifies the encoding fix: `find_source` with `AccountTrieShardedKey`
        // correctly resolves the root path even when child path [0] has history.
        // Before the length-prefix fix, this would return `FromCurrentState`
        // due to encoding ambiguity. Now it correctly returns `FromChangeset(15)`.
        let db = setup_db();
        let root_path = StoredNibbles(Nibbles::default());
        let child_path = StoredNibbles(Nibbles::from_nibbles([0]));

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut cursor = wtx.cursor_write::<V2AccountsTrieHistory>().expect("c");
            cursor
                .upsert(
                    AccountTrieShardedKey::new(root_path.clone(), u64::MAX),
                    &BlockNumberList::new_pre_sorted([10, 15]),
                )
                .expect("upsert root");
            cursor
                .upsert(
                    AccountTrieShardedKey::new(child_path, u64::MAX),
                    &BlockNumberList::new_pre_sorted([10, 15]),
                )
                .expect("upsert child");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cursor = tx.cursor_read::<V2AccountsTrieHistory>().expect("c");

        // find_source with AccountTrieShardedKey correctly returns FromChangeset(15)
        let result = find_source::<V2AccountsTrieHistory, _>(
            &mut cursor,
            AccountTrieShardedKey::new(root_path.clone(), 12),
            12,
            |k| k.key == root_path,
        )
        .expect("ok");

        assert_eq!(
            result,
            ResolvedSource::FromChangeset(15),
            "find_source with AccountTrieShardedKey should correctly resolve root path"
        );
    }

    // ====================== Account Cursor tests ======================

    #[test]
    fn account_cursor_reads_current_state_when_no_history() {
        let db = setup_db();
        let addr = B256::from([0xAA; 32]);
        let acc = sample_account(42);

        {
            let wtx = db.tx_mut().expect("rw tx");
            wtx.cursor_write::<V2HashedAccounts>()
                .expect("c")
                .upsert(addr, &acc)
                .expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2AccountCursor::new(
            tx.cursor_read::<V2HashedAccounts>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedAccountChangeSets>().expect("c"),
            u64::MAX,
            true,
        );

        let result = cur.seek(addr).expect("ok").expect("should find");
        assert_eq!(result.0, addr);
        assert_eq!(result.1.nonce, 42);
    }

    #[test]
    fn account_cursor_resolves_from_changeset_when_modified_after_target() {
        let db = setup_db();
        let addr = B256::from([0xBB; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");

            // Current state: nonce=10 (applied at block 5)
            wtx.cursor_write::<V2HashedAccounts>()
                .expect("c")
                .upsert(addr, &sample_account(10))
                .expect("upsert");

            // History bitmap: block 5 modified this account
            wtx.cursor_write::<V2HashedAccountsHistory>()
                .expect("c")
                .upsert(
                    HashedAccountShardedKey::new(addr, u64::MAX),
                    &BlockNumberList::new_pre_sorted([5]),
                )
                .expect("upsert");

            // Changeset: before block 5, account had nonce=3
            wtx.cursor_dup_write::<V2HashedAccountChangeSets>()
                .expect("c")
                .append_dup(5u64, HashedAccountBeforeTx::new(addr, Some(sample_account(3))))
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // Query at block 4 (before the modification at block 5)
        let mut cur = V2AccountCursor::new(
            tx.cursor_read::<V2HashedAccounts>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedAccountChangeSets>().expect("c"),
            4,
            false,
        );

        let result = cur.seek(addr).expect("ok").expect("should find");
        assert_eq!(result.0, addr);
        assert_eq!(result.1.nonce, 3, "should get changeset value (before block 5)");
    }

    #[test]
    fn account_cursor_returns_current_state_when_at_or_after_last_modification() {
        let db = setup_db();
        let addr = B256::from([0xCC; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");

            // Current state: nonce=20
            wtx.cursor_write::<V2HashedAccounts>()
                .expect("c")
                .upsert(addr, &sample_account(20))
                .expect("upsert");

            // History bitmap: [3, 7]
            wtx.cursor_write::<V2HashedAccountsHistory>()
                .expect("c")
                .upsert(
                    HashedAccountShardedKey::new(addr, u64::MAX),
                    &BlockNumberList::new_pre_sorted([3, 7]),
                )
                .expect("upsert");

            // Changeset at 3
            wtx.cursor_dup_write::<V2HashedAccountChangeSets>()
                .expect("c")
                .append_dup(3u64, HashedAccountBeforeTx::new(addr, Some(sample_account(1))))
                .expect("append");

            // Changeset at 7
            wtx.cursor_dup_write::<V2HashedAccountChangeSets>()
                .expect("c")
                .append_dup(7u64, HashedAccountBeforeTx::new(addr, Some(sample_account(5))))
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // Query at block 10 (after last modification at block 7)
        let mut cur = V2AccountCursor::new(
            tx.cursor_read::<V2HashedAccounts>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedAccountChangeSets>().expect("c"),
            10,
            true,
        );

        let result = cur.seek(addr).expect("ok").expect("should find");
        assert_eq!(result.0, addr);
        assert_eq!(result.1.nonce, 20, "current state (no modification after block 10)");
    }

    #[test]
    fn account_cursor_returns_none_when_not_yet_created() {
        let db = setup_db();
        let addr = B256::from([0xDD; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");

            // Current state: account exists (created at block 5)
            wtx.cursor_write::<V2HashedAccounts>()
                .expect("c")
                .upsert(addr, &sample_account(1))
                .expect("upsert");

            // History: first write at block 5
            wtx.cursor_write::<V2HashedAccountsHistory>()
                .expect("c")
                .upsert(
                    HashedAccountShardedKey::new(addr, u64::MAX),
                    &BlockNumberList::new_pre_sorted([5]),
                )
                .expect("upsert");

            // Changeset at 5: didn't exist before (info = None)
            wtx.cursor_dup_write::<V2HashedAccountChangeSets>()
                .expect("c")
                .append_dup(5u64, HashedAccountBeforeTx::new(addr, None))
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // Query at block 4 (before first write at 5)
        // The changeset at block 5 says info=None → the cursor's resolve returns None
        // → next_live_from skips this entry → seek returns None
        let mut cur = V2AccountCursor::new(
            tx.cursor_read::<V2HashedAccounts>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedAccountChangeSets>().expect("c"),
            4,
            false,
        );

        let result = cur.seek(addr).expect("ok");
        assert!(result.is_none(), "account should not exist at block 4");
    }

    #[test]
    fn account_cursor_seek_and_next_skip_dead_entries() {
        let db = setup_db();
        let k1 = B256::from([0x01; 32]);
        let k2 = B256::from([0x02; 32]);
        let k3 = B256::from([0x03; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut c = wtx.cursor_write::<V2HashedAccounts>().expect("c");
            c.upsert(k1, &sample_account(1)).expect("upsert");
            c.upsert(k2, &sample_account(2)).expect("upsert");
            c.upsert(k3, &sample_account(3)).expect("upsert");

            // k2 was created at block 10
            wtx.cursor_write::<V2HashedAccountsHistory>()
                .expect("c")
                .upsert(
                    HashedAccountShardedKey::new(k2, u64::MAX),
                    &BlockNumberList::new_pre_sorted([10]),
                )
                .expect("upsert");

            // Changeset at 10: k2 didn't exist before
            wtx.cursor_dup_write::<V2HashedAccountChangeSets>()
                .expect("c")
                .append_dup(10u64, HashedAccountBeforeTx::new(k2, None))
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // Query at block 5 (before k2 was created)
        let mut cur = V2AccountCursor::new(
            tx.cursor_read::<V2HashedAccounts>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedAccountChangeSets>().expect("c"),
            5,
            false,
        );

        // Seek k1 → should find k1 (no history = current state)
        let result = cur.seek(k1).expect("ok").expect("should find k1");
        assert_eq!(result.0, k1);
        assert_eq!(result.1.nonce, 1);

        // Next → should skip k2 (doesn't exist at block 5) and find k3
        let result = cur.next().expect("ok").expect("should skip k2, find k3");
        assert_eq!(result.0, k3);
        assert_eq!(result.1.nonce, 3);
    }

    /// Account was deleted (SELFDESTRUCT) after the target block, so it's not
    /// in the current-state table. The history walk must discover it.
    #[test]
    fn account_cursor_discovers_key_deleted_after_target_block() {
        let db = setup_db();
        let k1 = B256::from([0x01; 32]);
        let k2 = B256::from([0x02; 32]); // deleted after target
        let k3 = B256::from([0x03; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut c = wtx.cursor_write::<V2HashedAccounts>().expect("c");
            // k1 and k3 exist in current state; k2 was deleted at block 10
            c.upsert(k1, &sample_account(1)).expect("upsert");
            c.upsert(k3, &sample_account(3)).expect("upsert");

            // k2 history: modified at blocks [5, 10]
            wtx.cursor_write::<V2HashedAccountsHistory>()
                .expect("c")
                .upsert(
                    HashedAccountShardedKey::new(k2, u64::MAX),
                    &BlockNumberList::new_pre_sorted([5, 10]),
                )
                .expect("upsert");

            // Changeset at block 10: value before block 10 = nonce 7
            wtx.cursor_dup_write::<V2HashedAccountChangeSets>()
                .expect("c")
                .append_dup(10u64, HashedAccountBeforeTx::new(k2, Some(sample_account(7))))
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        // Query at block 9: k2 existed with nonce=7
        let mut cur = V2AccountCursor::new(
            tx.cursor_read::<V2HashedAccounts>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedAccountChangeSets>().expect("c"),
            9,
            false,
        );

        let r1 = cur.seek(B256::ZERO).expect("ok").expect("k1");
        assert_eq!(r1.0, k1);
        assert_eq!(r1.1.nonce, 1);

        let r2 = cur.next().expect("ok").expect("k2 from history");
        assert_eq!(r2.0, k2);
        assert_eq!(r2.1.nonce, 7);

        let r3 = cur.next().expect("ok").expect("k3");
        assert_eq!(r3.0, k3);
        assert_eq!(r3.1.nonce, 3);

        assert!(cur.next().expect("ok").is_none());
    }

    /// All accounts are deleted after the target block — only the history
    /// walk can find them.
    #[test]
    fn account_cursor_all_keys_from_history() {
        let db = setup_db();
        let k1 = B256::from([0x10; 32]);
        let k2 = B256::from([0x20; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            // Nothing in current state.

            // k1 modified at block 5
            wtx.cursor_write::<V2HashedAccountsHistory>()
                .expect("c")
                .upsert(
                    HashedAccountShardedKey::new(k1, u64::MAX),
                    &BlockNumberList::new_pre_sorted([5]),
                )
                .expect("upsert");
            wtx.cursor_dup_write::<V2HashedAccountChangeSets>()
                .expect("c")
                .append_dup(5u64, HashedAccountBeforeTx::new(k1, Some(sample_account(11))))
                .expect("append");

            // k2 modified at block 8
            wtx.cursor_write::<V2HashedAccountsHistory>()
                .expect("c")
                .upsert(
                    HashedAccountShardedKey::new(k2, u64::MAX),
                    &BlockNumberList::new_pre_sorted([8]),
                )
                .expect("upsert");
            wtx.cursor_dup_write::<V2HashedAccountChangeSets>()
                .expect("c")
                .append_dup(8u64, HashedAccountBeforeTx::new(k2, Some(sample_account(22))))
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2AccountCursor::new(
            tx.cursor_read::<V2HashedAccounts>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedAccountChangeSets>().expect("c"),
            4,
            false,
        );

        let r1 = cur.seek(B256::ZERO).expect("ok").expect("k1");
        assert_eq!(r1.0, k1);
        assert_eq!(r1.1.nonce, 11);

        let r2 = cur.next().expect("ok").expect("k2");
        assert_eq!(r2.0, k2);
        assert_eq!(r2.1.nonce, 22);

        assert!(cur.next().expect("ok").is_none());
    }

    /// Duplicate key in both current state and history — the merge should
    /// yield it exactly once.
    #[test]
    fn account_cursor_deduplicates_key_in_both_cursors() {
        let db = setup_db();
        let k = B256::from([0x55; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            wtx.cursor_write::<V2HashedAccounts>()
                .expect("c")
                .upsert(k, &sample_account(99))
                .expect("upsert");

            wtx.cursor_write::<V2HashedAccountsHistory>()
                .expect("c")
                .upsert(
                    HashedAccountShardedKey::new(k, u64::MAX),
                    &BlockNumberList::new_pre_sorted([5]),
                )
                .expect("upsert");
            wtx.cursor_dup_write::<V2HashedAccountChangeSets>()
                .expect("c")
                .append_dup(5u64, HashedAccountBeforeTx::new(k, Some(sample_account(50))))
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        // At block 4 → changeset at 5 gives nonce=50
        let mut cur = V2AccountCursor::new(
            tx.cursor_read::<V2HashedAccounts>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_read::<V2HashedAccountsHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedAccountChangeSets>().expect("c"),
            4,
            false,
        );

        let r = cur.seek(B256::ZERO).expect("ok").expect("one result");
        assert_eq!(r.0, k);
        assert_eq!(r.1.nonce, 50);
        assert!(cur.next().expect("ok").is_none(), "no duplicates");
    }

    // ====================== Account Trie Cursor tests ======================

    #[test]
    fn account_trie_cursor_reads_current_state_when_no_history() {
        let db = setup_db();
        let path = Nibbles::from_nibbles([0x0A]);
        let n = node();

        {
            let wtx = db.tx_mut().expect("rw tx");
            wtx.cursor_write::<V2AccountsTrie>()
                .expect("c")
                .upsert(StoredNibbles(path), &n)
                .expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            u64::MAX,
            true,
        );

        let out = TrieCursor::seek_exact(&mut cur, path).expect("ok").expect("some");
        assert_eq!(out.0, path);
        assert_eq!(out.1, n);
    }

    #[test]
    fn account_trie_cursor_resolves_old_node_from_changeset() {
        let db = setup_db();
        let path = Nibbles::from_nibbles([0x0B]);
        let old_node = node();
        let new_node = node2();

        {
            let wtx = db.tx_mut().expect("rw tx");

            // Current state has new_node (applied at block 10)
            wtx.cursor_write::<V2AccountsTrie>()
                .expect("c")
                .upsert(StoredNibbles(path), &new_node)
                .expect("upsert");

            // History: modified at block 10
            wtx.cursor_write::<V2AccountsTrieHistory>()
                .expect("c")
                .upsert(
                    AccountTrieShardedKey::new(StoredNibbles(path), u64::MAX),
                    &BlockNumberList::new_pre_sorted([10]),
                )
                .expect("upsert");

            // Changeset at block 10: old_node was the value before
            let cs_entry = TrieChangeSetsEntry {
                nibbles: StoredNibblesSubKey(path),
                node: Some(old_node.clone()),
            };
            wtx.cursor_dup_write::<V2AccountTrieChangeSets>()
                .expect("c")
                .append_dup(10u64, cs_entry)
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // Query at block 9 (before modification at 10)
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            9,
            false,
        );

        let out = TrieCursor::seek_exact(&mut cur, path).expect("ok").expect("some");
        assert_eq!(out.0, path);
        assert_eq!(out.1, old_node, "should get old node from changeset");
    }

    #[test]
    fn account_trie_cursor_seek_and_next_skip_dead_nodes() {
        let db = setup_db();
        let p1 = Nibbles::from_nibbles([0x01]);
        let p2 = Nibbles::from_nibbles([0x02]);
        let p3 = Nibbles::from_nibbles([0x03]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut c = wtx.cursor_write::<V2AccountsTrie>().expect("c");
            c.upsert(StoredNibbles(p1), &node()).expect("upsert");
            c.upsert(StoredNibbles(p2), &node()).expect("upsert");
            c.upsert(StoredNibbles(p3), &node()).expect("upsert");

            // p2 was created at block 5, didn't exist before
            wtx.cursor_write::<V2AccountsTrieHistory>()
                .expect("c")
                .upsert(
                    AccountTrieShardedKey::new(StoredNibbles(p2), u64::MAX),
                    &BlockNumberList::new_pre_sorted([5]),
                )
                .expect("upsert");

            let cs_entry = TrieChangeSetsEntry {
                nibbles: StoredNibblesSubKey(p2),
                node: None,
            };
            wtx.cursor_dup_write::<V2AccountTrieChangeSets>()
                .expect("c")
                .append_dup(5u64, cs_entry)
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // Query at block 3 (before p2 was created)
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            3,
            false,
        );

        // Seek p1 → should find p1
        let out = TrieCursor::seek(&mut cur, p1).expect("ok").expect("some");
        assert_eq!(out.0, p1);

        // Next → should skip p2 (didn't exist at block 3) and find p3
        let out = TrieCursor::next(&mut cur).expect("ok").expect("some");
        assert_eq!(out.0, p3, "should skip p2 which didn't exist at block 3");
    }

    #[test]
    fn account_trie_cursor_seek_returns_gte() {
        let db = setup_db();
        let p_a = Nibbles::from_nibbles([0x0A]);
        let p_c = Nibbles::from_nibbles([0x0C]);
        let p_b = Nibbles::from_nibbles([0x0B]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut c = wtx.cursor_write::<V2AccountsTrie>().expect("c");
            c.upsert(StoredNibbles(p_a), &node()).expect("upsert");
            c.upsert(StoredNibbles(p_c), &node()).expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            u64::MAX,
            true,
        );

        // Seek to 0x0B → should land on 0x0C (first ≥ 0x0B)
        let out = TrieCursor::seek(&mut cur, p_b).expect("ok").expect("some");
        assert_eq!(out.0, p_c);
    }

    #[test]
    fn account_trie_cursor_empty_returns_none() {
        let db = setup_db();
        let tx = db.tx().expect("ro tx");
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            u64::MAX,
            true,
        );

        assert!(TrieCursor::seek(&mut cur, Nibbles::default()).expect("ok").is_none());
        assert!(TrieCursor::next(&mut cur).expect("ok").is_none());
    }

    #[test]
    fn account_trie_cursor_discovers_key_deleted_after_target_block() {
        // Scenario from the design discussion:
        //   block 2 adds [a1, b1, c1]
        //   block 3 adds [d1, z1] and deletes a1
        // Query at block 2 → should see a1, b1, c1 (not d1 or z1)
        let db = setup_db();
        let a1 = Nibbles::from_nibbles([0x0A, 0x01]);
        let b1 = Nibbles::from_nibbles([0x0B, 0x01]);
        let c1 = Nibbles::from_nibbles([0x0C, 0x01]);
        let d1 = Nibbles::from_nibbles([0x0D, 0x01]);
        let z1 = Nibbles::from_nibbles([0x0F, 0x01]);
        let n = node();

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut c = wtx.cursor_write::<V2AccountsTrie>().expect("c");

            // Current state after block 3: {b1, c1, d1, z1} (a1 deleted)
            c.upsert(StoredNibbles(b1), &n).expect("upsert");
            c.upsert(StoredNibbles(c1), &n).expect("upsert");
            c.upsert(StoredNibbles(d1), &n).expect("upsert");
            c.upsert(StoredNibbles(z1), &n).expect("upsert");

            // History bitmaps
            let mut hc = wtx.cursor_write::<V2AccountsTrieHistory>().expect("c");
            // a1 modified at blocks 2 and 3
            hc.upsert(
                AccountTrieShardedKey::new(StoredNibbles(a1), u64::MAX),
                &BlockNumberList::new_pre_sorted([2, 3]),
            )
            .expect("upsert");
            // b1 modified at block 2
            hc.upsert(
                AccountTrieShardedKey::new(StoredNibbles(b1), u64::MAX),
                &BlockNumberList::new_pre_sorted([2]),
            )
            .expect("upsert");
            // c1 modified at block 2
            hc.upsert(
                AccountTrieShardedKey::new(StoredNibbles(c1), u64::MAX),
                &BlockNumberList::new_pre_sorted([2]),
            )
            .expect("upsert");
            // d1 modified at block 3
            hc.upsert(
                AccountTrieShardedKey::new(StoredNibbles(d1), u64::MAX),
                &BlockNumberList::new_pre_sorted([3]),
            )
            .expect("upsert");
            // z1 modified at block 3
            hc.upsert(
                AccountTrieShardedKey::new(StoredNibbles(z1), u64::MAX),
                &BlockNumberList::new_pre_sorted([3]),
            )
            .expect("upsert");

            // Changesets
            let mut csc = wtx.cursor_dup_write::<V2AccountTrieChangeSets>().expect("c");

            // Block 2 changesets: a1, b1, c1 didn't exist before
            csc.append_dup(
                2u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(a1), node: None },
            )
            .expect("append");
            csc.append_dup(
                2u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(b1), node: None },
            )
            .expect("append");
            csc.append_dup(
                2u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(c1), node: None },
            )
            .expect("append");

            // Block 3 changesets: a1 existed (deleted), d1 and z1 didn't exist
            csc.append_dup(
                3u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(a1), node: Some(n.clone()) },
            )
            .expect("append");
            csc.append_dup(
                3u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(d1), node: None },
            )
            .expect("append");
            csc.append_dup(
                3u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(z1), node: None },
            )
            .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // Query at block 2: a1 should be visible even though it's deleted
        // from current state (deleted at block 3).
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            2,
            false,
        );

        // seek(default) → a1 (discovered via history walk, resolved from changeset)
        let out = TrieCursor::seek(&mut cur, Nibbles::default())
            .expect("ok")
            .expect("should find a1");
        assert_eq!(out.0, a1, "a1 must be visible at block 2");
        assert_eq!(out.1, n);

        // next → b1
        let out = TrieCursor::next(&mut cur).expect("ok").expect("should find b1");
        assert_eq!(out.0, b1);

        // next → c1
        let out = TrieCursor::next(&mut cur).expect("ok").expect("should find c1");
        assert_eq!(out.0, c1);

        // next → None (d1 and z1 didn't exist at block 2)
        let out = TrieCursor::next(&mut cur).expect("ok");
        assert!(out.is_none(), "d1 and z1 must NOT be visible at block 2");
    }

    #[test]
    fn account_trie_cursor_deleted_key_only_in_history() {
        // Key exists ONLY in history (not in current state), no other keys at all.
        // Ensures the history-walk alone can produce results when current state is empty.
        let db = setup_db();
        let p = Nibbles::from_nibbles([0x05]);
        let n = node();

        {
            let wtx = db.tx_mut().expect("rw tx");
            // Current state: empty (p was deleted at block 4)

            // History: [2, 4]
            wtx.cursor_write::<V2AccountsTrieHistory>()
                .expect("c")
                .upsert(
                    AccountTrieShardedKey::new(StoredNibbles(p), u64::MAX),
                    &BlockNumberList::new_pre_sorted([2, 4]),
                )
                .expect("upsert");

            // Changeset block 2: p didn't exist before
            let mut csc = wtx.cursor_dup_write::<V2AccountTrieChangeSets>().expect("c");
            csc.append_dup(
                2u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(p), node: None },
            )
            .expect("append");
            // Changeset block 4: p had value n before deletion
            csc.append_dup(
                4u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(p), node: Some(n.clone()) },
            )
            .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // Query at block 3: p should be visible (created at 2, deleted at 4)
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            3,
            false,
        );

        let out = TrieCursor::seek(&mut cur, Nibbles::default())
            .expect("ok")
            .expect("should find p at block 3");
        assert_eq!(out.0, p);
        assert_eq!(out.1, n, "should resolve from changeset at block 4");

        // next → None
        assert!(TrieCursor::next(&mut cur).expect("ok").is_none());

        // Also: query at block 1 → p didn't exist yet
        let mut cur2 = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            1,
            false,
        );
        assert!(
            TrieCursor::seek(&mut cur2, Nibbles::default()).expect("ok").is_none(),
            "p should not exist at block 1"
        );
    }

    #[test]
    fn account_trie_cursor_seek_exact_on_deleted_key() {
        // seek_exact on a key that is deleted from current state but alive at
        // the target block.
        let db = setup_db();
        let p = Nibbles::from_nibbles([0x0A]);
        let n = node();

        {
            let wtx = db.tx_mut().expect("rw tx");
            // Current state: empty (p deleted at block 10)

            wtx.cursor_write::<V2AccountsTrieHistory>()
                .expect("c")
                .upsert(
                    AccountTrieShardedKey::new(StoredNibbles(p), u64::MAX),
                    &BlockNumberList::new_pre_sorted([5, 10]),
                )
                .expect("upsert");

            let mut csc = wtx.cursor_dup_write::<V2AccountTrieChangeSets>().expect("c");
            // Block 5: created (old = None)
            csc.append_dup(
                5u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(p), node: None },
            )
            .expect("append");
            // Block 10: deleted (old = n)
            csc.append_dup(
                10u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(p), node: Some(n.clone()) },
            )
            .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // seek_exact at block 8 → should find p
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            8,
            false,
        );
        let out = TrieCursor::seek_exact(&mut cur, p).expect("ok").expect("should find");
        assert_eq!(out.0, p);
        assert_eq!(out.1, n);

        // seek_exact at block 3 → should NOT find p (created at 5)
        let mut cur2 = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            3,
            false,
        );
        assert!(TrieCursor::seek_exact(&mut cur2, p).expect("ok").is_none());
    }

    #[test]
    fn account_trie_cursor_current_tracks_last_yielded() {
        // current() should return the last key yielded by seek/next.
        let db = setup_db();
        let p1 = Nibbles::from_nibbles([0x01]);
        let p2 = Nibbles::from_nibbles([0x02]);
        let n = node();

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut c = wtx.cursor_write::<V2AccountsTrie>().expect("c");
            c.upsert(StoredNibbles(p1), &n).expect("upsert");
            c.upsert(StoredNibbles(p2), &n).expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            u64::MAX,
            true,
        );

        // Before any seek, current is None
        assert!(TrieCursor::current(&mut cur).expect("ok").is_none());

        TrieCursor::seek(&mut cur, p1).expect("ok");
        assert_eq!(TrieCursor::current(&mut cur).expect("ok"), Some(p1));

        TrieCursor::next(&mut cur).expect("ok");
        assert_eq!(TrieCursor::current(&mut cur).expect("ok"), Some(p2));
    }

    #[test]
    fn account_trie_cursor_seek_exact_then_next() {
        // After seek_exact, next() should return the key after the sought key.
        let db = setup_db();
        let p1 = Nibbles::from_nibbles([0x01]);
        let p2 = Nibbles::from_nibbles([0x02]);
        let p3 = Nibbles::from_nibbles([0x03]);
        let n = node();

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut c = wtx.cursor_write::<V2AccountsTrie>().expect("c");
            c.upsert(StoredNibbles(p1), &n).expect("upsert");
            c.upsert(StoredNibbles(p2), &n).expect("upsert");
            c.upsert(StoredNibbles(p3), &n).expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            u64::MAX,
            true,
        );

        // seek_exact p2
        let out = TrieCursor::seek_exact(&mut cur, p2).expect("ok").expect("some");
        assert_eq!(out.0, p2);

        // next → p3
        let out = TrieCursor::next(&mut cur).expect("ok").expect("some");
        assert_eq!(out.0, p3);

        // next → None
        assert!(TrieCursor::next(&mut cur).expect("ok").is_none());
    }

    #[test]
    fn account_trie_cursor_seek_gte_skips_dead_landing() {
        // seek() lands on a dead key (in current state but not alive at target
        // block) and must skip forward to the next live key.
        let db = setup_db();
        let p_a = Nibbles::from_nibbles([0x0A]);
        let p_b = Nibbles::from_nibbles([0x0B]);
        let p_c = Nibbles::from_nibbles([0x0C]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut c = wtx.cursor_write::<V2AccountsTrie>().expect("c");
            c.upsert(StoredNibbles(p_a), &node()).expect("upsert");
            c.upsert(StoredNibbles(p_b), &node()).expect("upsert");
            c.upsert(StoredNibbles(p_c), &node()).expect("upsert");

            // p_b was created at block 10
            wtx.cursor_write::<V2AccountsTrieHistory>()
                .expect("c")
                .upsert(
                    AccountTrieShardedKey::new(StoredNibbles(p_b), u64::MAX),
                    &BlockNumberList::new_pre_sorted([10]),
                )
                .expect("upsert");

            wtx.cursor_dup_write::<V2AccountTrieChangeSets>()
                .expect("c")
                .append_dup(
                    10u64,
                    TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(p_b), node: None },
                )
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // At block 5, seek(p_b) → p_b is dead → should skip to p_c
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            5,
            false,
        );

        let out = TrieCursor::seek(&mut cur, p_b).expect("ok").expect("some");
        assert_eq!(out.0, p_c, "should skip dead p_b and land on p_c");
    }

    #[test]
    fn account_trie_cursor_all_keys_dead() {
        // Every key in current state is dead at the target block, and no
        // history-only keys exist → None.
        let db = setup_db();
        let p1 = Nibbles::from_nibbles([0x01]);
        let p2 = Nibbles::from_nibbles([0x02]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut c = wtx.cursor_write::<V2AccountsTrie>().expect("c");
            c.upsert(StoredNibbles(p1), &node()).expect("upsert");
            c.upsert(StoredNibbles(p2), &node()).expect("upsert");

            let mut hc = wtx.cursor_write::<V2AccountsTrieHistory>().expect("c");
            // Both created at block 5
            hc.upsert(
                AccountTrieShardedKey::new(StoredNibbles(p1), u64::MAX),
                &BlockNumberList::new_pre_sorted([5]),
            )
            .expect("upsert");
            hc.upsert(
                AccountTrieShardedKey::new(StoredNibbles(p2), u64::MAX),
                &BlockNumberList::new_pre_sorted([5]),
            )
            .expect("upsert");

            let mut csc = wtx.cursor_dup_write::<V2AccountTrieChangeSets>().expect("c");
            csc.append_dup(
                5u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(p1), node: None },
            )
            .expect("append");
            csc.append_dup(
                5u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(p2), node: None },
            )
            .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // At block 3 → both dead
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            3,
            false,
        );

        assert!(TrieCursor::seek(&mut cur, Nibbles::default()).expect("ok").is_none());
    }

    #[test]
    fn account_trie_cursor_interleaved_current_and_history_keys() {
        // Current state has {b, d}. History has {a, c, e} (all deleted after
        // target block). The merge should yield a, b, c, d, e in order.
        let db = setup_db();
        let a = Nibbles::from_nibbles([0x01]);
        let b = Nibbles::from_nibbles([0x02]);
        let c = Nibbles::from_nibbles([0x03]);
        let d = Nibbles::from_nibbles([0x04]);
        let e = Nibbles::from_nibbles([0x05]);
        let n = node();

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut cs = wtx.cursor_write::<V2AccountsTrie>().expect("c");
            // Current state: {b, d}
            cs.upsert(StoredNibbles(b), &n).expect("upsert");
            cs.upsert(StoredNibbles(d), &n).expect("upsert");

            let mut hc = wtx.cursor_write::<V2AccountsTrieHistory>().expect("c");
            // a: created block 2, deleted block 10
            hc.upsert(
                AccountTrieShardedKey::new(StoredNibbles(a), u64::MAX),
                &BlockNumberList::new_pre_sorted([2, 10]),
            )
            .expect("upsert");
            // b: created block 2 (stays in current state)
            hc.upsert(
                AccountTrieShardedKey::new(StoredNibbles(b), u64::MAX),
                &BlockNumberList::new_pre_sorted([2]),
            )
            .expect("upsert");
            // c: created block 2, deleted block 10
            hc.upsert(
                AccountTrieShardedKey::new(StoredNibbles(c), u64::MAX),
                &BlockNumberList::new_pre_sorted([2, 10]),
            )
            .expect("upsert");
            // d: created block 2 (stays)
            hc.upsert(
                AccountTrieShardedKey::new(StoredNibbles(d), u64::MAX),
                &BlockNumberList::new_pre_sorted([2]),
            )
            .expect("upsert");
            // e: created block 2, deleted block 10
            hc.upsert(
                AccountTrieShardedKey::new(StoredNibbles(e), u64::MAX),
                &BlockNumberList::new_pre_sorted([2, 10]),
            )
            .expect("upsert");

            let mut csc = wtx.cursor_dup_write::<V2AccountTrieChangeSets>().expect("c");
            // Block 2: all created (old = None)
            csc.append_dup(
                2u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(a), node: None },
            )
            .expect("append");
            csc.append_dup(
                2u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(b), node: None },
            )
            .expect("append");
            csc.append_dup(
                2u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(c), node: None },
            )
            .expect("append");
            csc.append_dup(
                2u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(d), node: None },
            )
            .expect("append");
            csc.append_dup(
                2u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(e), node: None },
            )
            .expect("append");
            // Block 10: a, c, e deleted (old = n)
            csc.append_dup(
                10u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(a), node: Some(n.clone()) },
            )
            .expect("append");
            csc.append_dup(
                10u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(c), node: Some(n.clone()) },
            )
            .expect("append");
            csc.append_dup(
                10u64,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(e), node: Some(n.clone()) },
            )
            .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // At block 5: a, b, c, d, e all alive (a, c, e via changeset at 10)
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            5,
            false,
        );

        let out = TrieCursor::seek(&mut cur, Nibbles::default()).expect("ok").expect("a");
        assert_eq!(out.0, a, "first key should be a (from history)");

        let out = TrieCursor::next(&mut cur).expect("ok").expect("b");
        assert_eq!(out.0, b, "second key should be b (from current state)");

        let out = TrieCursor::next(&mut cur).expect("ok").expect("c");
        assert_eq!(out.0, c, "third key should be c (from history)");

        let out = TrieCursor::next(&mut cur).expect("ok").expect("d");
        assert_eq!(out.0, d, "fourth key should be d (from current state)");

        let out = TrieCursor::next(&mut cur).expect("ok").expect("e");
        assert_eq!(out.0, e, "fifth key should be e (from history)");

        assert!(TrieCursor::next(&mut cur).expect("ok").is_none());
    }

    #[test]
    fn account_trie_cursor_duplicate_key_in_both_cursors() {
        // Key exists in BOTH current state and history. The merge should NOT
        // yield it twice.
        let db = setup_db();
        let p = Nibbles::from_nibbles([0x0A]);
        let n = node();
        let n2 = node2();

        {
            let wtx = db.tx_mut().expect("rw tx");

            // Current state: p -> n2 (updated at block 10)
            wtx.cursor_write::<V2AccountsTrie>()
                .expect("c")
                .upsert(StoredNibbles(p), &n2)
                .expect("upsert");

            // History: modified at block 10
            wtx.cursor_write::<V2AccountsTrieHistory>()
                .expect("c")
                .upsert(
                    AccountTrieShardedKey::new(StoredNibbles(p), u64::MAX),
                    &BlockNumberList::new_pre_sorted([10]),
                )
                .expect("upsert");

            wtx.cursor_dup_write::<V2AccountTrieChangeSets>()
                .expect("c")
                .append_dup(
                    10u64,
                    TrieChangeSetsEntry {
                        nibbles: StoredNibblesSubKey(p),
                        node: Some(n.clone()),
                    },
                )
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // At block 8: resolve from changeset at 10 → old value n
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            8,
            false,
        );

        let out = TrieCursor::seek(&mut cur, p).expect("ok").expect("should find");
        assert_eq!(out.0, p);
        assert_eq!(out.1, n, "should get old value from changeset");

        // next → None (should NOT yield p again)
        assert!(TrieCursor::next(&mut cur).expect("ok").is_none());
    }

    #[test]
    fn account_trie_cursor_query_at_latest_block() {
        // When max_block_number == u64::MAX, everything reads from current
        // state — even keys with history. This exercises the
        // FromCurrentState path in find_source.
        let db = setup_db();
        let p1 = Nibbles::from_nibbles([0x01]);
        let p2 = Nibbles::from_nibbles([0x02]);
        let n2 = node2();

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut c = wtx.cursor_write::<V2AccountsTrie>().expect("c");
            c.upsert(StoredNibbles(p1), &node()).expect("upsert");
            c.upsert(StoredNibbles(p2), &n2).expect("upsert");

            // p2 has history at block 5
            wtx.cursor_write::<V2AccountsTrieHistory>()
                .expect("c")
                .upsert(
                    AccountTrieShardedKey::new(StoredNibbles(p2), u64::MAX),
                    &BlockNumberList::new_pre_sorted([5]),
                )
                .expect("upsert");

            wtx.cursor_dup_write::<V2AccountTrieChangeSets>()
                .expect("c")
                .append_dup(
                    5u64,
                    TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(p2), node: None },
                )
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            u64::MAX,
            true,
        );

        let out = TrieCursor::seek(&mut cur, Nibbles::default()).expect("ok").expect("p1");
        assert_eq!(out.0, p1);

        let out = TrieCursor::next(&mut cur).expect("ok").expect("p2");
        assert_eq!(out.0, p2);
        assert_eq!(out.1, n2, "should read current state value");

        assert!(TrieCursor::next(&mut cur).expect("ok").is_none());
    }

    // ——— Storage Trie Cursor: dual-cursor merge tests ———

    #[test]
    fn storage_trie_cursor_discovers_deleted_key() {
        // Same scenario as account trie deleted-key test, but for storage trie.
        let db = setup_db();
        let addr = B256::from([0xAA; 32]);
        let a1 = Nibbles::from_nibbles([0x0A, 0x01]);
        let b1 = Nibbles::from_nibbles([0x0B, 0x01]);
        let c1 = Nibbles::from_nibbles([0x0C, 0x01]);
        let n = node();

        {
            let wtx = db.tx_mut().expect("rw tx");

            // Current state: {b1, c1} (a1 deleted at block 5)
            let mut sc = wtx.cursor_dup_write::<V2StoragesTrie>().expect("c");
            sc.upsert(
                addr,
                &StorageTrieEntry { nibbles: StoredNibblesSubKey(b1), node: n.clone() },
            )
            .expect("upsert");
            sc.upsert(
                addr,
                &StorageTrieEntry { nibbles: StoredNibblesSubKey(c1), node: n.clone() },
            )
            .expect("upsert");

            let mut hc = wtx.cursor_write::<V2StoragesTrieHistory>().expect("c");
            // a1: created at block 2, deleted at block 5
            hc.upsert(
                StorageTrieShardedKey::new(addr, StoredNibbles(a1), u64::MAX),
                &BlockNumberList::new_pre_sorted([2, 5]),
            )
            .expect("upsert");
            // b1: created at block 2
            hc.upsert(
                StorageTrieShardedKey::new(addr, StoredNibbles(b1), u64::MAX),
                &BlockNumberList::new_pre_sorted([2]),
            )
            .expect("upsert");
            // c1: created at block 2
            hc.upsert(
                StorageTrieShardedKey::new(addr, StoredNibbles(c1), u64::MAX),
                &BlockNumberList::new_pre_sorted([2]),
            )
            .expect("upsert");

            let mut csc = wtx.cursor_dup_write::<V2StorageTrieChangeSets>().expect("c");
            // Block 2: all created
            let cs_key2 = BlockNumberHashedAddress((2u64, addr));
            csc.append_dup(
                cs_key2,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(a1), node: None },
            )
            .expect("append");
            csc.append_dup(
                cs_key2,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(b1), node: None },
            )
            .expect("append");
            csc.append_dup(
                cs_key2,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(c1), node: None },
            )
            .expect("append");
            // Block 5: a1 deleted (old = n)
            let cs_key5 = BlockNumberHashedAddress((5u64, addr));
            csc.append_dup(
                cs_key5,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(a1), node: Some(n) },
            )
            .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // Query at block 3: a1 should be visible
        let mut cur = V2StorageTrieCursor::new(
            tx.cursor_dup_read::<V2StoragesTrie>().expect("c"),
            tx.cursor_read::<V2StoragesTrieHistory>().expect("c"),
            tx.cursor_read::<V2StoragesTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2StorageTrieChangeSets>().expect("c"),
            addr,
            3,
            false,
        );

        let out = TrieCursor::seek(&mut cur, Nibbles::default())
            .expect("ok")
            .expect("should find a1");
        assert_eq!(out.0, a1, "a1 must be visible at block 3");

        let out = TrieCursor::next(&mut cur).expect("ok").expect("b1");
        assert_eq!(out.0, b1);

        let out = TrieCursor::next(&mut cur).expect("ok").expect("c1");
        assert_eq!(out.0, c1);

        assert!(TrieCursor::next(&mut cur).expect("ok").is_none());
    }

    #[test]
    fn storage_trie_cursor_deleted_key_does_not_cross_address() {
        // Deleted history key from addr_b must NOT appear when walking addr_a.
        let db = setup_db();
        let addr_a = B256::from([0x11; 32]);
        let addr_b = B256::from([0x22; 32]);
        let p1 = Nibbles::from_nibbles([0x01]);
        let p2 = Nibbles::from_nibbles([0x02]);
        let n = node();

        {
            let wtx = db.tx_mut().expect("rw tx");

            // Current state: addr_a has {p1}, addr_b is empty (p2 deleted)
            let mut sc = wtx.cursor_dup_write::<V2StoragesTrie>().expect("c");
            sc.upsert(
                addr_a,
                &StorageTrieEntry { nibbles: StoredNibblesSubKey(p1), node: n.clone() },
            )
            .expect("upsert");

            // addr_b: p2 history (created block 2, deleted block 5)
            wtx.cursor_write::<V2StoragesTrieHistory>()
                .expect("c")
                .upsert(
                    StorageTrieShardedKey::new(addr_b, StoredNibbles(p2), u64::MAX),
                    &BlockNumberList::new_pre_sorted([2, 5]),
                )
                .expect("upsert");

            let mut csc = wtx.cursor_dup_write::<V2StorageTrieChangeSets>().expect("c");
            csc.append_dup(
                BlockNumberHashedAddress((2u64, addr_b)),
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(p2), node: None },
            )
            .expect("append");
            csc.append_dup(
                BlockNumberHashedAddress((5u64, addr_b)),
                TrieChangeSetsEntry {
                    nibbles: StoredNibblesSubKey(p2),
                    node: Some(n),
                },
            )
            .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // Walk addr_a at block 3 → only p1
        let mut cur = V2StorageTrieCursor::new(
            tx.cursor_dup_read::<V2StoragesTrie>().expect("c"),
            tx.cursor_read::<V2StoragesTrieHistory>().expect("c"),
            tx.cursor_read::<V2StoragesTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2StorageTrieChangeSets>().expect("c"),
            addr_a,
            3,
            true,
        );

        let out = TrieCursor::seek(&mut cur, Nibbles::default()).expect("ok").expect("p1");
        assert_eq!(out.0, p1);
        assert!(
            TrieCursor::next(&mut cur).expect("ok").is_none(),
            "must not leak addr_b's history into addr_a"
        );
    }

    #[test]
    fn storage_trie_cursor_set_hashed_address_resets_merge_state() {
        // After set_hashed_address, the merge state must be reset so seek/next
        // operate correctly on the new address.
        let db = setup_db();
        let addr_a = B256::from([0x55; 32]);
        let addr_b = B256::from([0x66; 32]);
        let p1 = Nibbles::from_nibbles([0x01]);
        let p2 = Nibbles::from_nibbles([0x02]);
        let n = node();

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut c = wtx.cursor_dup_write::<V2StoragesTrie>().expect("c");
            c.upsert(
                addr_a,
                &StorageTrieEntry { nibbles: StoredNibblesSubKey(p1), node: n.clone() },
            )
            .expect("upsert");
            c.upsert(
                addr_b,
                &StorageTrieEntry { nibbles: StoredNibblesSubKey(p2), node: n },
            )
            .expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2StorageTrieCursor::new(
            tx.cursor_dup_read::<V2StoragesTrie>().expect("c"),
            tx.cursor_read::<V2StoragesTrieHistory>().expect("c"),
            tx.cursor_read::<V2StoragesTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2StorageTrieChangeSets>().expect("c"),
            addr_a,
            u64::MAX,
            true,
        );

        // Seek on addr_a
        let out = TrieCursor::seek(&mut cur, p1).expect("ok").expect("p1");
        assert_eq!(out.0, p1);

        // Switch to addr_b
        cur.set_hashed_address(addr_b);
        let out = TrieCursor::seek(&mut cur, p2).expect("ok").expect("p2");
        assert_eq!(out.0, p2);

        assert!(TrieCursor::next(&mut cur).expect("ok").is_none());
    }

    // ====================== Storage Cursor tests ======================

    #[test]
    fn storage_cursor_reads_current_state_when_no_history() {
        let db = setup_db();
        let addr = B256::from([0xAA; 32]);
        let slot = B256::from([0x01; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            wtx.cursor_dup_write::<V2HashedStorages>()
                .expect("c")
                .upsert(addr, &StorageEntry { key: slot, value: U256::from(42u64) })
                .expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2StorageCursor::new(
            tx.cursor_dup_read::<V2HashedStorages>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedStorageChangeSets>().expect("c"),
            addr,
            u64::MAX,
            true,
        );

        let result = cur.seek(slot).expect("ok").expect("should find");
        assert_eq!(result, (slot, U256::from(42u64)));
    }

    #[test]
    fn storage_cursor_resolves_from_changeset() {
        let db = setup_db();
        let addr = B256::from([0xAA; 32]);
        let slot = B256::from([0x01; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");

            // Current state: value=1000
            wtx.cursor_dup_write::<V2HashedStorages>()
                .expect("c")
                .upsert(addr, &StorageEntry { key: slot, value: U256::from(1000u64) })
                .expect("upsert");

            // History: modified at block 8
            wtx.cursor_write::<V2HashedStoragesHistory>()
                .expect("c")
                .upsert(
                    HashedStorageShardedKey {
                        hashed_address: addr,
                        sharded_key: ShardedKey::new(slot, u64::MAX),
                    },
                    &BlockNumberList::new_pre_sorted([8]),
                )
                .expect("upsert");

            // Changeset at block 8: old value was 500
            let cs_key = BlockNumberHashedAddress((8u64, addr));
            wtx.cursor_dup_write::<V2HashedStorageChangeSets>()
                .expect("c")
                .append_dup(cs_key, StorageEntry { key: slot, value: U256::from(500u64) })
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // Query at block 7 (before modification at 8)
        let mut cur = V2StorageCursor::new(
            tx.cursor_dup_read::<V2HashedStorages>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedStorageChangeSets>().expect("c"),
            addr,
            7,
            false,
        );

        let result = cur.seek(slot).expect("ok").expect("should find");
        assert_eq!(result.0, slot);
        assert_eq!(result.1, U256::from(500u64), "should get changeset value");
    }

    #[test]
    fn storage_cursor_is_storage_empty() {
        let db = setup_db();
        let addr_with = B256::from([0xBB; 32]);
        let addr_without = B256::from([0xCC; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            wtx.cursor_dup_write::<V2HashedStorages>()
                .expect("c")
                .upsert(
                    addr_with,
                    &StorageEntry { key: B256::from([0x01; 32]), value: U256::from(1u64) },
                )
                .expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        let mut cur_with = V2StorageCursor::new(
            tx.cursor_dup_read::<V2HashedStorages>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedStorageChangeSets>().expect("c"),
            addr_with,
            u64::MAX,
            true,
        );
        assert!(!cur_with.is_storage_empty().expect("ok"));

        let mut cur_without = V2StorageCursor::new(
            tx.cursor_dup_read::<V2HashedStorages>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedStorageChangeSets>().expect("c"),
            addr_without,
            u64::MAX,
            true,
        );
        assert!(cur_without.is_storage_empty().expect("ok"));
    }

    /// Storage slot was zeroed (deleted from `V2HashedStorages`) after the target
    /// block. The history walk must discover it.
    #[test]
    fn storage_cursor_discovers_slot_deleted_after_target_block() {
        let db = setup_db();
        let addr = B256::from([0xAA; 32]);
        let s1 = B256::from([0x01; 32]);
        let s2 = B256::from([0x02; 32]); // deleted after target
        let s3 = B256::from([0x03; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut c = wtx.cursor_dup_write::<V2HashedStorages>().expect("c");
            // s1 and s3 exist; s2 was zeroed at block 10
            c.upsert(addr, &StorageEntry { key: s1, value: U256::from(100u64) })
                .expect("upsert");
            c.upsert(addr, &StorageEntry { key: s3, value: U256::from(300u64) })
                .expect("upsert");

            // s2 history: modified at [5, 10]
            wtx.cursor_write::<V2HashedStoragesHistory>()
                .expect("c")
                .upsert(
                    HashedStorageShardedKey {
                        hashed_address: addr,
                        sharded_key: ShardedKey::new(s2, u64::MAX),
                    },
                    &BlockNumberList::new_pre_sorted([5, 10]),
                )
                .expect("upsert");

            // Changeset at block 10: s2 = 200 before block 10
            let cs_key = BlockNumberHashedAddress((10u64, addr));
            wtx.cursor_dup_write::<V2HashedStorageChangeSets>()
                .expect("c")
                .append_dup(cs_key, StorageEntry { key: s2, value: U256::from(200u64) })
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2StorageCursor::new(
            tx.cursor_dup_read::<V2HashedStorages>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedStorageChangeSets>().expect("c"),
            addr,
            9,
            false,
        );

        let r1 = cur.seek(B256::ZERO).expect("ok").expect("s1");
        assert_eq!(r1.0, s1);
        assert_eq!(r1.1, U256::from(100u64));

        let r2 = cur.next().expect("ok").expect("s2 from history");
        assert_eq!(r2.0, s2);
        assert_eq!(r2.1, U256::from(200u64));

        let r3 = cur.next().expect("ok").expect("s3");
        assert_eq!(r3.0, s3);
        assert_eq!(r3.1, U256::from(300u64));

        assert!(cur.next().expect("ok").is_none());
    }

    /// `is_storage_empty` must return false when storage existed at the target
    /// block but has since been wiped from current state.
    #[test]
    fn storage_cursor_is_storage_empty_false_for_historical_only_slots() {
        let db = setup_db();
        let addr = B256::from([0xDD; 32]);
        let slot = B256::from([0x01; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            // No current state for addr — all storage was wiped at block 10.

            // History: slot modified at [5, 10]
            wtx.cursor_write::<V2HashedStoragesHistory>()
                .expect("c")
                .upsert(
                    HashedStorageShardedKey {
                        hashed_address: addr,
                        sharded_key: ShardedKey::new(slot, u64::MAX),
                    },
                    &BlockNumberList::new_pre_sorted([5, 10]),
                )
                .expect("upsert");

            // Changeset at 10: value=42 before block 10
            let cs_key = BlockNumberHashedAddress((10u64, addr));
            wtx.cursor_dup_write::<V2HashedStorageChangeSets>()
                .expect("c")
                .append_dup(cs_key, StorageEntry { key: slot, value: U256::from(42u64) })
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2StorageCursor::new(
            tx.cursor_dup_read::<V2HashedStorages>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedStorageChangeSets>().expect("c"),
            addr,
            9,
            false,
        );

        assert!(!cur.is_storage_empty().expect("ok"), "should find historical slot");
    }

    /// History-only storage key must NOT cross into a different address.
    #[test]
    fn storage_cursor_deleted_slot_does_not_cross_address() {
        let db = setup_db();
        let addr1 = B256::from([0x01; 32]);
        let addr2 = B256::from([0x02; 32]);
        let slot = B256::from([0x0A; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            // No current storage for addr1.
            // addr2 has a history entry for the same slot.
            wtx.cursor_write::<V2HashedStoragesHistory>()
                .expect("c")
                .upsert(
                    HashedStorageShardedKey {
                        hashed_address: addr2,
                        sharded_key: ShardedKey::new(slot, u64::MAX),
                    },
                    &BlockNumberList::new_pre_sorted([5]),
                )
                .expect("upsert");

            let cs_key = BlockNumberHashedAddress((5u64, addr2));
            wtx.cursor_dup_write::<V2HashedStorageChangeSets>()
                .expect("c")
                .append_dup(cs_key, StorageEntry { key: slot, value: U256::from(99u64) })
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2StorageCursor::new(
            tx.cursor_dup_read::<V2HashedStorages>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedStorageChangeSets>().expect("c"),
            addr1,
            4,
            true,
        );

        // addr1 has no storage — history walk finds addr2's entry but must filter it out.
        assert!(cur.seek(B256::ZERO).expect("ok").is_none());
    }

    /// `set_hashed_address` resets merge state so the cursor works correctly
    /// for the new address.
    #[test]
    fn storage_cursor_set_hashed_address_resets_merge_state() {
        let db = setup_db();
        let addr1 = B256::from([0x01; 32]);
        let addr2 = B256::from([0x02; 32]);
        let slot = B256::from([0x0A; 32]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut c = wtx.cursor_dup_write::<V2HashedStorages>().expect("c");
            c.upsert(addr1, &StorageEntry { key: slot, value: U256::from(11u64) })
                .expect("upsert");
            c.upsert(addr2, &StorageEntry { key: slot, value: U256::from(22u64) })
                .expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2StorageCursor::new(
            tx.cursor_dup_read::<V2HashedStorages>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_read::<V2HashedStoragesHistory>().expect("c"),
            tx.cursor_dup_read::<V2HashedStorageChangeSets>().expect("c"),
            addr1,
            u64::MAX,
            true,
        );

        let r1 = cur.seek(B256::ZERO).expect("ok").expect("addr1 slot");
        assert_eq!(r1.1, U256::from(11u64));

        cur.set_hashed_address(addr2);
        let r2 = cur.seek(B256::ZERO).expect("ok").expect("addr2 slot");
        assert_eq!(r2.1, U256::from(22u64));
    }

    // Regression: root trie path [] with child path [0] history.
    // Before the length-prefixed encoding fix, the V2AccountTrieCursor would return
    // the current-state root node instead of the historical one.
    #[test]
    fn account_trie_cursor_root_path_resolves_historical_with_child_paths() {
        let db = setup_db();
        let root_path = Nibbles::default();
        let child_path = Nibbles::from_nibbles([0]);

        let root_node_at_block5 = BranchNodeCompact::new(
            0b11,
            0,
            0,
            vec![],
            Some(B256::repeat_byte(0x55)),
        );
        let root_node_at_block10 = BranchNodeCompact::new(
            0b111,
            0,
            0,
            vec![],
            Some(B256::repeat_byte(0xAA)),
        );
        let child_node_at_block10 = BranchNodeCompact::new(
            0b101,
            0,
            0,
            vec![],
            Some(B256::repeat_byte(0xBB)),
        );

        {
            let wtx = db.tx_mut().expect("rw tx");

            // Current state: root has block 10's node, child has block 10's node
            wtx.cursor_write::<V2AccountsTrie>()
                .expect("c")
                .upsert(StoredNibbles(root_path), &root_node_at_block10)
                .expect("upsert root");
            wtx.cursor_write::<V2AccountsTrie>()
                .expect("c")
                .upsert(StoredNibbles(child_path), &child_node_at_block10)
                .expect("upsert child");

            // Changeset at block 10: root had block5's node before block 10
            wtx.cursor_dup_write::<V2AccountTrieChangeSets>()
                .expect("c")
                .append_dup(
                    10,
                    TrieChangeSetsEntry {
                        nibbles: StoredNibblesSubKey(root_path),
                        node: Some(root_node_at_block5.clone()),
                    },
                )
                .expect("append root cs");

            // History: root modified at block 10
            wtx.cursor_write::<V2AccountsTrieHistory>()
                .expect("c")
                .upsert(
                    AccountTrieShardedKey::new(StoredNibbles(root_path), u64::MAX),
                    &BlockNumberList::new_pre_sorted([10]),
                )
                .expect("upsert root history");

            // History: child [0] modified at block 10
            wtx.cursor_write::<V2AccountsTrieHistory>()
                .expect("c")
                .upsert(
                    AccountTrieShardedKey::new(StoredNibbles(child_path), u64::MAX),
                    &BlockNumberList::new_pre_sorted([10]),
                )
                .expect("upsert child history");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2AccountTrieCursor::new(
            tx.cursor_read::<V2AccountsTrie>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_read::<V2AccountsTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2AccountTrieChangeSets>().expect("c"),
            8,     // max_block_number: query at block 8 (before block 10's change)
            false, // is_latest = false (historical query)
        );

        // seek_exact on root path should return the historical value (block 5's node)
        let out = TrieCursor::seek_exact(&mut cur, root_path)
            .expect("ok")
            .expect("root should exist at block 8");
        assert_eq!(
            out.1, root_node_at_block5,
            "Root path should return the historical node from changeset, not current state"
        );
    }

    // ====================== Storage Trie Cursor tests ======================

    #[test]
    fn storage_trie_cursor_reads_current_state_when_no_history() {
        let db = setup_db();
        let addr = B256::from([0x55; 32]);
        let path = Nibbles::from_nibbles([0x0D]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            wtx.cursor_dup_write::<V2StoragesTrie>()
                .expect("c")
                .upsert(
                    addr,
                    &StorageTrieEntry { nibbles: StoredNibblesSubKey(path), node: node() },
                )
                .expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2StorageTrieCursor::new(
            tx.cursor_dup_read::<V2StoragesTrie>().expect("c"),
            tx.cursor_read::<V2StoragesTrieHistory>().expect("c"),
            tx.cursor_read::<V2StoragesTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2StorageTrieChangeSets>().expect("c"),
            addr,
            u64::MAX,
            true,
        );

        let out = TrieCursor::seek_exact(&mut cur, path).expect("ok").expect("some");
        assert_eq!(out.0, path);
    }

    #[test]
    fn storage_trie_cursor_resolves_from_changeset() {
        let db = setup_db();
        let addr = B256::from([0x55; 32]);
        let path = Nibbles::from_nibbles([0x0D]);
        let old_node = node();
        let new_node = node2();

        {
            let wtx = db.tx_mut().expect("rw tx");

            // Current state
            wtx.cursor_dup_write::<V2StoragesTrie>()
                .expect("c")
                .upsert(
                    addr,
                    &StorageTrieEntry { nibbles: StoredNibblesSubKey(path), node: new_node },
                )
                .expect("upsert");

            // History: modified at block 6
            wtx.cursor_write::<V2StoragesTrieHistory>()
                .expect("c")
                .upsert(
                    StorageTrieShardedKey::new(addr, StoredNibbles(path), u64::MAX),
                    &BlockNumberList::new_pre_sorted([6]),
                )
                .expect("upsert");

            // Changeset at block 6: old node
            let cs_key = BlockNumberHashedAddress((6u64, addr));
            let cs_entry = TrieChangeSetsEntry {
                nibbles: StoredNibblesSubKey(path),
                node: Some(old_node.clone()),
            };
            wtx.cursor_dup_write::<V2StorageTrieChangeSets>()
                .expect("c")
                .append_dup(cs_key, cs_entry)
                .expect("append");

            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");

        // Query at block 5 (before modification at 6)
        let mut cur = V2StorageTrieCursor::new(
            tx.cursor_dup_read::<V2StoragesTrie>().expect("c"),
            tx.cursor_read::<V2StoragesTrieHistory>().expect("c"),
            tx.cursor_read::<V2StoragesTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2StorageTrieChangeSets>().expect("c"),
            addr,
            5,
            false,
        );

        let out = TrieCursor::seek_exact(&mut cur, path).expect("ok").expect("some");
        assert_eq!(out.0, path);
        assert_eq!(out.1, old_node, "should get old node from changeset");
    }

    #[test]
    fn storage_trie_cursor_respects_address_boundary() {
        let db = setup_db();
        let addr_a = B256::from([0x33; 32]);
        let addr_b = B256::from([0x44; 32]);
        let p1 = Nibbles::from_nibbles([0x05]);
        let p2 = Nibbles::from_nibbles([0x06]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut c = wtx.cursor_dup_write::<V2StoragesTrie>().expect("c");
            c.upsert(
                addr_a,
                &StorageTrieEntry { nibbles: StoredNibblesSubKey(p1), node: node() },
            )
            .expect("upsert");
            c.upsert(
                addr_b,
                &StorageTrieEntry { nibbles: StoredNibblesSubKey(p2), node: node() },
            )
            .expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2StorageTrieCursor::new(
            tx.cursor_dup_read::<V2StoragesTrie>().expect("c"),
            tx.cursor_read::<V2StoragesTrieHistory>().expect("c"),
            tx.cursor_read::<V2StoragesTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2StorageTrieChangeSets>().expect("c"),
            addr_a,
            u64::MAX,
            true,
        );

        let out = TrieCursor::seek(&mut cur, p1).expect("ok").expect("some");
        assert_eq!(out.0, p1);

        // next() should return None — crossed address boundary
        let out = TrieCursor::next(&mut cur).expect("ok");
        assert!(out.is_none(), "must not cross address boundary (DupSort)");
    }

    #[test]
    fn storage_trie_cursor_set_hashed_address() {
        let db = setup_db();
        let addr_a = B256::from([0x55; 32]);
        let addr_b = B256::from([0x66; 32]);
        let path = Nibbles::from_nibbles([0x01]);

        {
            let wtx = db.tx_mut().expect("rw tx");
            let mut c = wtx.cursor_dup_write::<V2StoragesTrie>().expect("c");
            c.upsert(
                addr_a,
                &StorageTrieEntry { nibbles: StoredNibblesSubKey(path), node: node() },
            )
            .expect("upsert");
            c.upsert(
                addr_b,
                &StorageTrieEntry { nibbles: StoredNibblesSubKey(path), node: node() },
            )
            .expect("upsert");
            wtx.commit().expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = V2StorageTrieCursor::new(
            tx.cursor_dup_read::<V2StoragesTrie>().expect("c"),
            tx.cursor_read::<V2StoragesTrieHistory>().expect("c"),
            tx.cursor_read::<V2StoragesTrieHistory>().expect("c"),
            tx.cursor_dup_read::<V2StorageTrieChangeSets>().expect("c"),
            addr_a,
            u64::MAX,
            true,
        );

        assert!(TrieCursor::seek_exact(&mut cur, path).expect("ok").is_some());
        cur.set_hashed_address(addr_b);
        assert!(TrieCursor::seek_exact(&mut cur, path).expect("ok").is_some());
    }
}
