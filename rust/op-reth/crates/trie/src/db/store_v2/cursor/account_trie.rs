//! History-aware cursor over the [`V2AccountsTrie`] v2 table.

use reth_db::{
    DatabaseError,
    cursor::{DbCursorRO, DbDupCursorRO},
};
use reth_trie::{
    BranchNodeCompact, Nibbles, StoredNibbles, StoredNibblesSubKey, trie_cursor::TrieCursor,
};

use super::{MergeState, resolve_historical};
use crate::db::models::{
    AccountTrieShardedKey, V2AccountTrieChangeSets, V2AccountsTrie, V2AccountsTrieHistory,
};

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
    /// Shared merge-walk state.
    state: MergeState<StoredNibbles, (StoredNibbles, BranchNodeCompact)>,
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
            state: MergeState::new(),
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
        let target = path.clone();
        let max_block_number = self.max_block_number;
        let hc = &mut self.history_cursor;
        let cc = &mut self.changeset_cursor;
        let cur = &mut self.cursor;
        resolve_historical::<V2AccountsTrieHistory, _, _>(
            hc,
            max_block_number,
            |bn| AccountTrieShardedKey::new(target.clone(), bn),
            |k| k.key == target,
            |block| {
                Ok(cc
                    .seek_by_key_subkey(block, StoredNibblesSubKey(target.0))?
                    .filter(|e| e.nibbles == StoredNibblesSubKey(target.0))
                    .and_then(|e| e.node))
            },
            || Ok(cur.seek_exact(target.clone())?.map(|(_, node)| node)),
        )
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
        let target = path.clone();
        let max_block_number = self.max_block_number;
        let hc = &mut self.history_cursor;
        let cc = &mut self.changeset_cursor;
        resolve_historical::<V2AccountsTrieHistory, _, _>(
            hc,
            max_block_number,
            |bn| AccountTrieShardedKey::new(target.clone(), bn),
            |k| k.key == target,
            |block| {
                Ok(cc
                    .seek_by_key_subkey(block, StoredNibblesSubKey(target.0))?
                    .filter(|e| e.nibbles == StoredNibblesSubKey(target.0))
                    .and_then(|e| e.node))
            },
            || Ok(cs_value.cloned()),
        )
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
    fn find_next_live(&mut self) -> Result<Option<(Nibbles, BranchNodeCompact)>, DatabaseError> {
        loop {
            // Step 1: Pick the minimum key from current-state and history cursors.
            // If both have the same key, prefer the current-state value.
            // `cs_value` is `Some` when the key exists in current state, `None`
            // when it only appears in history (i.e. deleted after max_block_number).
            let (min_key, cs_value) = match (&self.state.cs_next, &self.state.hist_next_key) {
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

            // Step 2: Advance whichever cursor(s) produced this key.
            // Both are advanced when they have the same key (deduplication).
            if self.state.cs_next.as_ref().is_some_and(|(k, _)| *k == min_key) {
                self.state.cs_next = self.cursor.next()?;
            }
            if self.state.hist_next_key.as_ref().is_some_and(|k| *k == min_key) {
                self.state.hist_next_key = self.advance_history_past(&min_key)?;
            }

            // Step 3: Resolve the value at max_block_number.
            // Returns `Some` if the key was live at that block, `None` if it
            // didn't exist yet or was already deleted.
            if let Some(node) = self.resolve_node_merge(&min_key, cs_value.as_ref())? {
                self.state.last_key = Some(min_key.clone());
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
    CC: DbCursorRO<V2AccountTrieChangeSets> + DbDupCursorRO<V2AccountTrieChangeSets> + Send,
{
    fn seek_exact(
        &mut self,
        key: Nibbles,
    ) -> Result<Option<(Nibbles, BranchNodeCompact)>, DatabaseError> {
        self.state.seeked = true;

        if self.is_latest {
            // Fast path: direct current-state lookup.
            let result = self.cursor.seek_exact(StoredNibbles(key))?;
            if result.is_some() {
                self.state.last_key = Some(StoredNibbles(key));
            }
            return Ok(result.map(|(_, node)| (key, node)));
        }

        let path = StoredNibbles(key);
        let node = self.resolve_node_standalone(&path)?;

        // Re-sync the walk state so a subsequent next() starts after `path`.
        let cs_at_key = self.cursor.seek(path.clone())?;
        self.state.cs_next = match cs_at_key {
            Some((k, _)) if k == path => self.cursor.next()?,
            other => other,
        };
        self.state.hist_next_key = self.advance_history_past(&path)?;

        if node.is_some() {
            self.state.last_key = Some(path);
        }
        Ok(node.map(|n| (key, n)))
    }

    fn seek(
        &mut self,
        key: Nibbles,
    ) -> Result<Option<(Nibbles, BranchNodeCompact)>, DatabaseError> {
        self.state.seeked = true;

        if self.is_latest {
            // Fast path: direct current-state walk.
            let result = self.cursor.seek(StoredNibbles(key))?;
            if let Some((ref k, _)) = result {
                self.state.last_key = Some(k.clone());
            }
            return Ok(result.map(|(k, node)| (k.0, node)));
        }

        // Initialize both merge cursors at the target key.
        self.state.cs_next = self.cursor.seek(StoredNibbles(key))?;
        self.state.hist_next_key = self
            .history_walk_cursor
            .seek(AccountTrieShardedKey::new(StoredNibbles(key), 0))?
            .map(|(k, _)| k.key);
        self.find_next_live()
    }

    fn next(&mut self) -> Result<Option<(Nibbles, BranchNodeCompact)>, DatabaseError> {
        if !self.state.seeked {
            return self.seek(Nibbles::default());
        }

        if self.is_latest {
            let result = self.cursor.next()?;
            if let Some((ref k, _)) = result {
                self.state.last_key = Some(k.clone());
            }
            return Ok(result.map(|(k, node)| (k.0, node)));
        }

        self.find_next_live()
    }

    fn current(&mut self) -> Result<Option<Nibbles>, DatabaseError> {
        Ok(self.state.last_key.as_ref().map(|k| k.0))
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}
