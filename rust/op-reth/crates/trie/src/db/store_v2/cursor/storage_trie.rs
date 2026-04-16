//! History-aware cursor over the [`V2StoragesTrie`] v2 `DupSort` table.

use alloy_primitives::B256;
use reth_db::{
    DatabaseError,
    cursor::{DbCursorRO, DbDupCursorRO},
};
use reth_trie::{
    BranchNodeCompact, Nibbles, StoredNibbles, StoredNibblesSubKey,
    trie_cursor::{TrieCursor, TrieStorageCursor},
};
use reth_trie_common::StorageTrieEntry;

use super::{ResolvedSource, find_source};
use crate::db::models::{
    BlockNumberHashedAddress, StorageTrieShardedKey, V2StorageTrieChangeSets, V2StoragesTrie,
    V2StoragesTrieHistory,
};

/// History-aware cursor over the [`V2StoragesTrie`] v2 `DupSort` table.
///
/// Uses the same dual-cursor merge strategy as [`super::V2AccountTrieCursor`] but
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
            self.max_block_number.saturating_add(1),
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
            self.max_block_number.saturating_add(1),
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
        let seek = StorageTrieShardedKey::new(self.hashed_address, key.clone(), u64::MAX);
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
    fn find_next_live(&mut self) -> Result<Option<(Nibbles, BranchNodeCompact)>, DatabaseError> {
        loop {
            // Step 1: Pick the minimum key from current-state and history cursors.
            // If both have the same key, prefer the current-state value.
            // `cs_node` is `Some` when the key exists in current state, `None`
            // when it only appears in history (i.e. deleted after max_block_number).
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

            // Step 2: Advance whichever cursor(s) produced this key.
            // Both are advanced when they have the same key (deduplication).
            if self.cs_next.as_ref().is_some_and(|e| StoredNibbles(e.nibbles.0) == min_nibbles) {
                self.cs_next = self.cursor.next_dup()?.map(|(_, v)| v);
            }
            if self.hist_next_key.as_ref().is_some_and(|k| *k == min_nibbles) {
                self.hist_next_key = self.advance_history_past(&min_nibbles)?;
            }

            // Step 3: Resolve the value at max_block_number.
            // Returns `Some` if the key was live at that block, `None` if it
            // didn't exist yet or was already deleted.
            if let Some(node) = self.resolve_node_merge(min_nibbles.0, cs_node.as_ref())? {
                self.last_key = Some(StoredNibbles(min_nibbles.0));
                return Ok(Some((min_nibbles.0, node)));
            }
            // Key doesn't exist at max_block_number — continue to next.
        }
    }
}

impl<C, HC, CC> TrieCursor for V2StorageTrieCursor<C, HC, CC>
where
    C: DbCursorRO<V2StoragesTrie> + DbDupCursorRO<V2StoragesTrie> + Send,
    HC: DbCursorRO<V2StoragesTrieHistory> + Send,
    CC: DbCursorRO<V2StorageTrieChangeSets> + DbDupCursorRO<V2StorageTrieChangeSets> + Send,
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
        let hist_seek = StorageTrieShardedKey::new(self.hashed_address, StoredNibbles(key), 0);
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

    fn reset(&mut self) {
        self.cs_next = None;
        self.hist_next_key = None;
        self.last_key = None;
        self.seeked = false;
    }
}

impl<C, HC, CC> TrieStorageCursor for V2StorageTrieCursor<C, HC, CC>
where
    C: DbCursorRO<V2StoragesTrie> + DbDupCursorRO<V2StoragesTrie> + Send,
    HC: DbCursorRO<V2StoragesTrieHistory> + Send,
    CC: DbCursorRO<V2StorageTrieChangeSets> + DbDupCursorRO<V2StorageTrieChangeSets> + Send,
{
    fn set_hashed_address(&mut self, hashed_address: B256) {
        self.hashed_address = hashed_address;
        self.cs_next = None;
        self.hist_next_key = None;
        self.last_key = None;
        self.seeked = false;
    }
}
