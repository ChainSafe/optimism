//! V2 cursor implementations for the v2 table schema.
//!
//! These cursors implement **history-aware reads** using the v2 3-table-per-data-type pattern:
//!
//! | Purpose | Accounts | Storages | Account Trie | Storage Trie |
//! |---------|----------|----------|-------------|-------------|
//! | Current state | `V2HashedAccounts` | `V2HashedStorages` | `V2AccountsTrie` | `V2StoragesTrie` |
//! | `ChangeSets` | `V2HashedAccountChangeSets` | `V2HashedStorageChangeSets` | `V2AccountTrieChangeSets` | `V2StorageTrieChangeSets` |
//! | History | `V2HashedAccountsHistory` | `V2HashedStoragesHistory` | `V2AccountsTrieHistory` | `V2StoragesTrieHistory` |
//!
//! # Historical Lookup Strategy
//!
//! Each cursor accepts a `max_block_number` parameter. For each key encountered:
//!
//! 1. **History bitmap lookup**: Seek `ShardedKey(key, max_block_number + 1)` in the history table.
//!    This lands on the first shard whose `highest_block_number > max_block_number`. The bitmap
//!    tells us which blocks modified this key.
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
/// **Important**: `seek_key` must embed `max_block_number + 1` (not
/// `max_block_number` itself) so that the seek lands on a shard whose
/// `highest_block_number > max_block_number`. This guarantees the shard
/// contains at least one entry after the target block, eliminating the
/// need for a `cursor.next()` fallback.
///
/// The algorithm:
/// 1. Seek the first history shard with `highest_block_number > max_block_number` (achieved by
///    embedding `max_block_number + 1` in `seek_key`).
/// 2. Within that shard, find the first block strictly `> max_block_number`.
/// 3. If found → `FromChangeset(block)`.
/// 4. Otherwise → `FromCurrentState`.
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
    // 1. Seek using the caller-provided key (which embeds max_block_number + 1), then filter to
    //    ensure the shard belongs to the expected key.
    let shard = cursor.seek(seek_key)?.filter(|(k, _)| key_filter(k));
    let Some((_, chunk)) = shard else {
        return Ok(ResolvedSource::FromCurrentState);
    };

    // 2. rank(n) = count of entries ≤ n. select(rank) = first entry > n.
    let rank = chunk.rank(max_block_number);
    Ok(chunk
        .select(rank)
        .map(ResolvedSource::FromChangeset)
        .unwrap_or(ResolvedSource::FromCurrentState))
}

#[cfg(test)]
mod tests;
