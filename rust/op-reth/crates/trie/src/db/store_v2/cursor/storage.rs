//! History-aware cursor over the [`V2HashedStorages`] v2 `DupSort` table.

use alloy_primitives::{B256, U256};
use reth_db::{
    DatabaseError,
    cursor::{DbCursorRO, DbDupCursorRO},
    models::sharded_key::ShardedKey,
};
use reth_trie::hashed_cursor::{HashedCursor, HashedStorageCursor};

use super::{ResolvedSource, find_source};
use crate::db::models::{
    BlockNumberHashedAddress, HashedStorageShardedKey, V2HashedStorageChangeSets, V2HashedStorages,
    V2HashedStoragesHistory,
};

/// History-aware cursor over the [`V2HashedStorages`] v2 `DupSort` table.
///
/// Uses the same dual-cursor merge strategy as [`super::V2AccountCursor`] but
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
            ResolvedSource::FromCurrentState => Ok(cs_value.copied().filter(|v| !v.is_zero())),
        }
    }

    /// Advance the history walk cursor past all shards of `key` (for this
    /// address) and return the next distinct storage key, if any.
    fn advance_history_past(&mut self, key: &B256) -> Result<Option<B256>, DatabaseError> {
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
    fn find_next_live(&mut self) -> Result<Option<(B256, U256)>, DatabaseError> {
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
    CC: DbCursorRO<V2HashedStorageChangeSets> + DbDupCursorRO<V2HashedStorageChangeSets> + Send,
{
    type Value = U256;

    fn seek(&mut self, subkey: B256) -> Result<Option<(B256, Self::Value)>, DatabaseError> {
        self.seeked = true;

        if self.is_latest {
            // Fast path: current state is authoritative.
            // Loop to skip zero-valued entries (tombstones).
            let mut entry = self.cursor.seek_by_key_subkey(self.hashed_address, subkey)?;
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
    CC: DbCursorRO<V2HashedStorageChangeSets> + DbDupCursorRO<V2HashedStorageChangeSets> + Send,
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
