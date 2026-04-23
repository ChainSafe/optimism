//! History-aware cursor over the [`V2HashedAccounts`] v2 tables.

use alloy_primitives::B256;
use reth_db::{
    DatabaseError,
    cursor::{DbCursorRO, DbDupCursorRO},
};
use reth_primitives_traits::Account;
use reth_trie::hashed_cursor::HashedCursor;

use super::resolve_historical;
use crate::db::models::{
    HashedAccountShardedKey, V2HashedAccountChangeSets, V2HashedAccounts, V2HashedAccountsHistory,
};

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
        let max_block_number = self.max_block_number;
        let hc = &mut self.history_cursor;
        let cc = &mut self.changeset_cursor;
        resolve_historical::<V2HashedAccountsHistory, _, _>(
            hc,
            max_block_number,
            |bn| HashedAccountShardedKey::new(hashed_address, bn),
            |k| k.0.key == hashed_address,
            |block| Ok(cc
                .seek_by_key_subkey(block, hashed_address)?
                .filter(|e| e.hashed_address == hashed_address)
                .and_then(|e| e.info)),
            || Ok(cs_value.copied()),
        )
    }

    /// Advance the history walk cursor past all shards of `key` and return
    /// the next distinct key, if any.
    fn advance_history_past(&mut self, key: &B256) -> Result<Option<B256>, DatabaseError> {
        let entry = self.history_walk_cursor.seek(HashedAccountShardedKey::new(*key, u64::MAX))?;
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
    fn find_next_live(&mut self) -> Result<Option<(B256, Account)>, DatabaseError> {
        loop {
            // Step 1: Pick the minimum key from current-state and history cursors.
            // If both have the same key, prefer the current-state value.
            // `cs_value` is `Some` when the key exists in current state, `None`
            // when it only appears in history (i.e. deleted after max_block_number).
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

            // Step 2: Advance whichever cursor(s) produced this key.
            // Both are advanced when they have the same key (deduplication).
            if self.cs_next.as_ref().is_some_and(|(k, _)| *k == min_key) {
                self.cs_next = self.cursor.next()?;
            }
            if self.hist_next_key.as_ref().is_some_and(|k| *k == min_key) {
                self.hist_next_key = self.advance_history_past(&min_key)?;
            }

            // Step 3: Resolve the value at max_block_number.
            // Returns `Some` if the key was live at that block, `None` if it
            // didn't exist yet or was already deleted.
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
