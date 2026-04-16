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
//! 1. **History bitmap lookup**: Seek `ShardedKey(key, max_block_number)` in the history table. The
//!    bitmap tells us which blocks modified this key.
//! 2. **Find the first modification *after* `max_block_number`**: Using `rank` + `select` on the
//!    bitmap. `rank(max_block_number)` counts entries ≤ the target block; `select(rank)` returns
//!    the first entry strictly greater.
//! 3. **Determine where the value lives**:
//!    - If a block `> max_block_number` modified this key → read the **changeset** at that block.
//!      The changeset stores the value *before* that block's execution, which is the value at the
//!      end of `max_block_number`.
//!    - If no block after `max_block_number` modified this key → the **current state** table
//!      already has the correct value.

mod account;
mod account_trie;
mod storage;
mod storage_trie;

pub use account::V2AccountCursor;
pub use account_trie::V2AccountTrieCursor;
pub use storage::V2StorageCursor;
pub use storage_trie::V2StorageTrieCursor;

use reth_db::{BlockNumberList, DatabaseError, cursor::DbCursorRO, table::Table};

/// Enum to define where to read the value for a given key at a specific block.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum ResolvedSource {
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
/// 4. If the shard boundary was hit (all entries ≤ `max_block_number`), advance to the next shard
///    for the same key. If found → use its first entry.
/// 5. Otherwise → `FromCurrentState`.
pub(crate) fn find_source<T, C>(
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

    // 3. All entries in this shard are ≤ max_block_number (shard boundary hit). The next shard (if
    //    it exists for the same key) starts after this one.
    if let Some((_, next_chunk)) = cursor.next()?.filter(|(k, _)| key_filter(k)) &&
        let Some(block) = next_chunk.select(0)
    {
        return Ok(ResolvedSource::FromChangeset(block));
    }

    Ok(ResolvedSource::FromCurrentState)
}

#[cfg(test)]
mod tests;
