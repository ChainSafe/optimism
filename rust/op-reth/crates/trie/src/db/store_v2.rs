//! V2 MDBX implementation of [`OpProofsStore`](crate::OpProofsStore).
//!
//! This module implements the v2 table schema using reth's **3-table-per-data-type** pattern:
//!
//! | Domain | Current State | ChangeSet | History Bitmap |
//! |--------|--------------|-----------|----------------|
//! | Hashed Accounts | [`HashedAccounts`] | [`HashedAccountChangeSets`] | [`HashedAccountsHistory`] |
//! | Hashed Storages | [`HashedStorages`] | [`HashedStorageChangeSets`] | [`HashedStoragesHistory`] |
//! | Account Trie | [`AccountsTrie`] | [`AccountTrieChangeSets`] | [`AccountsTrieHistory`] |
//! | Storage Trie | [`StoragesTrie`] | [`StorageTrieChangeSets`] | [`StoragesTrieHistory`] |
//!
//! # Write Flow
//!
//! When storing trie updates for a new block:
//! 1. Read old values from **current state** tables.
//! 2. Write old values to **changeset** tables (keyed by block number).
//! 3. Append block number to **history bitmap** tables (sharded).
//! 4. Overwrite **current state** tables with new values.
//!
//! # Prune Flow
//!
//! Delete changeset and history entries for pruned block ranges. Current state is unchanged.
//!
//! # Unwind Flow
//!
//! Restore old values from changesets back into current state, then delete the changeset/history
//! entries for the unwound block range.

use super::{BlockNumberHash, ProofWindow, ProofWindowKey, Tables};
use crate::{
    api::{
        InitialStateAnchor, InitialStateStatus, OpProofsInitProvider, OpProofsProviderRO,
        OpProofsProviderRw, OpProofsStore, WriteCounts,
    },
    db::{
        cursor_v2::{V2AccountCursor, V2AccountTrieCursor, V2StorageCursor, V2StorageTrieCursor},
        models::{
            AccountTrieShardedKey, AccountsTrie, AccountTrieChangeSets,
            AccountsTrieHistory, BlockNumberHashedAddress, HashedAccountBeforeTx,
            HashedAccountChangeSets, HashedAccountShardedKey, HashedAccounts,
            HashedAccountsHistory, HashedStorageChangeSets, HashedStorageShardedKey,
            HashedStorages, HashedStoragesHistory, StorageTrieShardedKey, StoragesTrie,
            StorageTrieChangeSets, StoragesTrieHistory, TrieChangeSetsEntry,
        },
        HashedStorageKey, StorageTrieKey,
    },
    BlockStateDiff, OpProofsStorageError, OpProofsStorageResult,
};
use alloy_eips::{eip1898::BlockWithParent, BlockNumHash, NumHash};
use alloy_primitives::{BlockNumber, B256, U256};
#[cfg(feature = "metrics")]
use metrics::{gauge, Label};
use reth_db::{
    cursor::{DbCursorRO, DbCursorRW, DbDupCursorRO, DbDupCursorRW},
    models::sharded_key::ShardedKey,
    mdbx::{init_db_for, DatabaseArguments},
    table::Table,
    transaction::{DbTx, DbTxMut},
    BlockNumberList, Database, DatabaseEnv, DatabaseError,
};
use reth_primitives_traits::{Account, StorageEntry};
use reth_trie::{
    updates::{StorageTrieUpdates, TrieUpdates},
    BranchNodeCompact, HashedPostState, Nibbles, StorageTrieEntry, StoredNibbles,
    StoredNibblesSubKey,
};
use std::{collections::{BTreeMap, BTreeSet}, fmt::Debug, path::Path, sync::Arc};

/// Maximum number of block indices per shard in history bitmap tables.
///
/// Matches reth's `NUM_OF_INDICES_IN_SHARD`.
const NUM_OF_INDICES_IN_SHARD: usize = 2_000;

// =============================================================================
// Storage (Database Environment)
// =============================================================================

/// V2 MDBX implementation of [`OpProofsStore`].
///
/// Uses the v2 3-table-per-data-type schema. Each data domain (accounts, storages,
/// account trie, storage trie) has a current-state table, a changeset table,
/// and a sharded history bitmap table.
#[derive(Debug)]
pub struct MdbxProofsStorageV2 {
    env: DatabaseEnv,
}

impl MdbxProofsStorageV2 {
    /// Creates a new [`MdbxProofsStorageV2`] instance with the given path.
    pub fn new(path: &Path) -> Result<Self, OpProofsStorageError> {
        let env = init_db_for::<_, Tables>(path, DatabaseArguments::default())
            .map_err(|e| DatabaseError::Other(format!("Failed to open database: {e}")))?;
        Ok(Self { env })
    }
}

impl OpProofsStore for MdbxProofsStorageV2 {
    type ProviderRO<'a> = Arc<MdbxProofsProviderV2<<DatabaseEnv as Database>::TX>>;
    type ProviderRw<'a> = MdbxProofsProviderV2<<DatabaseEnv as Database>::TXMut>;
    type Initializer<'a> = MdbxProofsProviderV2<<DatabaseEnv as Database>::TXMut>;

    fn provider_ro<'a>(&'a self) -> OpProofsStorageResult<Self::ProviderRO<'a>> {
        Ok(Arc::new(MdbxProofsProviderV2::new(self.env.tx()?)))
    }

    fn provider_rw<'a>(&'a self) -> OpProofsStorageResult<Self::ProviderRw<'a>> {
        Ok(MdbxProofsProviderV2::new(self.env.tx_mut()?))
    }

    fn initialization_provider<'a>(&'a self) -> OpProofsStorageResult<Self::Initializer<'a>> {
        Ok(MdbxProofsProviderV2::new(self.env.tx_mut()?))
    }
}

/// [`DatabaseMetrics`](reth_db::database_metrics::DatabaseMetrics) implementation for
/// [`MdbxProofsStorageV2`]. Reports per-table size, page counts, and entry counts.
#[cfg(feature = "metrics")]
impl reth_db::database_metrics::DatabaseMetrics for MdbxProofsStorageV2 {
    fn report_metrics(&self) {
        for (name, value, labels) in self.gauge_metrics() {
            gauge!(name, labels).set(value);
        }
    }

    fn gauge_metrics(&self) -> Vec<(&'static str, f64, Vec<Label>)> {
        use eyre::WrapErr;
        use tracing::error;

        let mut metrics = Vec::new();

        let _ = self
            .env
            .view(|tx| {
                for table in Tables::ALL.iter().map(Tables::name) {
                    let table_db = tx.inner().open_db(Some(table)).wrap_err("Could not open db.")?;

                    let stats = tx
                        .inner()
                        .db_stat(table_db.dbi())
                        .wrap_err(format!("Could not find table: {table}"))?;

                    let page_size = stats.page_size() as usize;
                    let leaf_pages = stats.leaf_pages();
                    let branch_pages = stats.branch_pages();
                    let overflow_pages = stats.overflow_pages();
                    let num_pages = leaf_pages + branch_pages + overflow_pages;
                    let table_size = page_size * num_pages;
                    let entries = stats.entries();

                    metrics.push((
                        "optimism_proof_storage.table_size",
                        table_size as f64,
                        vec![Label::new("table", table)],
                    ));
                    metrics.push((
                        "optimism_proof_storage.table_pages",
                        leaf_pages as f64,
                        vec![Label::new("table", table), Label::new("type", "leaf")],
                    ));
                    metrics.push((
                        "optimism_proof_storage.table_pages",
                        branch_pages as f64,
                        vec![Label::new("table", table), Label::new("type", "branch")],
                    ));
                    metrics.push((
                        "optimism_proof_storage.table_pages",
                        overflow_pages as f64,
                        vec![Label::new("table", table), Label::new("type", "overflow")],
                    ));
                    metrics.push((
                        "optimism_proof_storage.table_entries",
                        entries as f64,
                        vec![Label::new("table", table)],
                    ));
                }

                Ok::<(), eyre::Report>(())
            })
            .map_err(|error| error!(%error, "Failed to read db table stats"));

        if let Ok(freelist) =
            self.env.freelist().map_err(|error| error!(%error, "Failed to read db.freelist"))
        {
            metrics.push(("optimism_proof_storage.freelist", freelist as f64, vec![]));
        }

        if let Ok(stat) =
            self.env.stat().map_err(|error| error!(%error, "Failed to read db.stat"))
        {
            metrics.push(("optimism_proof_storage.page_size", stat.page_size() as f64, vec![]));
        }

        metrics.push((
            "optimism_proof_storage.timed_out_not_aborted_transactions",
            self.env.timed_out_not_aborted_transactions() as f64,
            vec![],
        ));

        metrics
    }
}

// =============================================================================
// Provider (Transaction wrapper)
// =============================================================================

/// V2 MDBX provider for proof storage, wrapping a database transaction.
#[derive(Debug)]
pub struct MdbxProofsProviderV2<TX> {
    tx: TX,
}

impl<TX> MdbxProofsProviderV2<TX> {
    /// Creates a new [`MdbxProofsProviderV2`].
    pub fn new(tx: TX) -> Self {
        Self { tx }
    }
}

// =============================================================================
// Read-only helpers
// =============================================================================

impl<TX: DbTx> MdbxProofsProviderV2<TX> {
    fn get_block_number_hash_inner(
        &self,
        key: ProofWindowKey,
    ) -> OpProofsStorageResult<Option<(u64, B256)>> {
        let mut cursor = self.tx.cursor_read::<ProofWindow>()?;
        Ok(cursor.seek_exact(key)?.map(|(_, val)| (val.number(), *val.hash())))
    }

    fn get_latest_block_number_hash_inner(&self) -> OpProofsStorageResult<Option<(u64, B256)>> {
        let block = self.get_block_number_hash_inner(ProofWindowKey::LatestBlock)?;
        if block.is_some() {
            return Ok(block);
        }
        self.get_block_number_hash_inner(ProofWindowKey::EarliestBlock)
    }

    /// Returns `true` when `max_block_number` is >= the latest stored block,
    /// meaning the current-state tables are authoritative and history/changeset
    /// lookups can be skipped entirely.
    fn is_latest_block(&self, max_block_number: u64) -> OpProofsStorageResult<bool> {
        match self.get_latest_block_number_hash_inner()? {
            Some((latest, _)) => Ok(max_block_number >= latest),
            // No blocks stored yet → current state is empty but correct.
            None => Ok(true),
        }
    }

    fn get_proof_window_inner(
        &self,
    ) -> OpProofsStorageResult<Option<(NumHash, NumHash)>> {
        let mut cursor = self.tx.cursor_read::<ProofWindow>()?;

        let earliest = match cursor.seek_exact(ProofWindowKey::EarliestBlock)? {
            Some((_, val)) => NumHash::new(val.number(), *val.hash()),
            None => return Ok(None),
        };

        let latest = match cursor.seek_exact(ProofWindowKey::LatestBlock)? {
            Some((_, val)) => NumHash::new(val.number(), *val.hash()),
            None => earliest,
        };

        Ok(Some((earliest, latest)))
    }

    fn get_initial_state_anchor_inner(&self) -> OpProofsStorageResult<Option<BlockNumHash>> {
        let mut cur = self.tx.cursor_read::<ProofWindow>()?;
        Ok(cur.seek_exact(ProofWindowKey::InitialStateAnchor)?.map(|(_k, v)| v.into()))
    }

    /// Fetch the state diff for a block from changeset tables.
    fn fetch_trie_updates_inner(
        &self,
        block_number: u64,
    ) -> OpProofsStorageResult<BlockStateDiff> {
        let mut trie_updates = TrieUpdates::default();

        // Account trie changesets
        {
            let mut cs_cursor = self.tx.cursor_read::<AccountTrieChangeSets>()?;
            let mut walker = cs_cursor.walk(Some(block_number))?;
            while let Some(Ok((bn, entry))) = walker.next() {
                if bn != block_number {
                    break;
                }
                let path = entry.nibbles.0;
                let current_node = self
                    .tx
                    .cursor_read::<AccountsTrie>()?
                    .seek_exact(StoredNibbles(path))?
                    .map(|(_, node)| node);

                match current_node {
                    Some(node) => {
                        trie_updates.account_nodes.insert(path, node);
                    }
                    None => {
                        trie_updates.removed_nodes.insert(path);
                    }
                }
            }
        }

        // Storage trie changesets
        {
            let mut cs_cursor = self.tx.cursor_read::<StorageTrieChangeSets>()?;
            let start = BlockNumberHashedAddress((block_number, B256::ZERO));
            let end = BlockNumberHashedAddress((block_number, B256::repeat_byte(0xff)));
            let mut walker = cs_cursor.walk_range(start..=end)?;

            while let Some(Ok((key, entry))) = walker.next() {
                let hashed_address = key.0 .1;
                let path = entry.nibbles.0;

                let current_node = self
                    .tx
                    .cursor_dup_read::<StoragesTrie>()?
                    .seek_by_key_subkey(hashed_address, StoredNibblesSubKey(path))?
                    .filter(|e| e.nibbles == StoredNibblesSubKey(path))
                    .map(|e| e.node);

                let stu = trie_updates
                    .storage_tries
                    .entry(hashed_address)
                    .or_insert_with(StorageTrieUpdates::default);

                match current_node {
                    Some(node) => {
                        stu.storage_nodes.insert(path, node);
                    }
                    None => {
                        stu.removed_nodes.insert(path);
                    }
                }
            }
        }

        // Hashed account changesets
        let mut post_state = HashedPostState::default();
        {
            let mut cs_cursor = self.tx.cursor_read::<HashedAccountChangeSets>()?;
            let mut walker = cs_cursor.walk(Some(block_number))?;
            while let Some(Ok((bn, entry))) = walker.next() {
                if bn != block_number {
                    break;
                }
                let current_account = self
                    .tx
                    .cursor_read::<HashedAccounts>()?
                    .seek_exact(entry.hashed_address)?
                    .map(|(_, acc)| acc);

                post_state.accounts.insert(entry.hashed_address, current_account);
            }
        }

        // Hashed storage changesets
        {
            let mut cs_cursor = self.tx.cursor_read::<HashedStorageChangeSets>()?;
            let start = BlockNumberHashedAddress((block_number, B256::ZERO));
            let end = BlockNumberHashedAddress((block_number, B256::repeat_byte(0xff)));
            let mut walker = cs_cursor.walk_range(start..=end)?;

            while let Some(Ok((key, entry))) = walker.next() {
                let hashed_address = key.0 .1;

                let current_value = self
                    .tx
                    .cursor_dup_read::<HashedStorages>()?
                    .seek_by_key_subkey(hashed_address, entry.key)?
                    .filter(|e| e.key == entry.key)
                    .map(|e| e.value)
                    .unwrap_or(U256::ZERO);

                let hs = post_state.storages.entry(hashed_address).or_default();
                hs.storage.insert(entry.key, current_value);
            }
        }

        Ok(BlockStateDiff {
            sorted_trie_updates: trie_updates.into_sorted(),
            sorted_post_state: post_state.into_sorted(),
        })
    }
}

// =============================================================================
// Read-write helpers
// =============================================================================

impl<TX: DbTxMut + DbTx> MdbxProofsProviderV2<TX> {
    fn set_earliest_block_number_inner(
        &self,
        block_number: u64,
        hash: B256,
    ) -> OpProofsStorageResult<()> {
        let mut cursor = self.tx.cursor_write::<ProofWindow>()?;
        cursor.upsert(ProofWindowKey::EarliestBlock, &BlockNumberHash::new(block_number, hash))?;
        Ok(())
    }

    fn set_latest_block_number_inner(
        &self,
        block_number: u64,
        hash: B256,
    ) -> OpProofsStorageResult<()> {
        let mut cursor = self.tx.cursor_write::<ProofWindow>()?;
        cursor.upsert(ProofWindowKey::LatestBlock, &BlockNumberHash::new(block_number, hash))?;
        Ok(())
    }

    // -------------------------------------------------------------------------
    // History bitmap helpers (sharded)
    // -------------------------------------------------------------------------

    /// Append a block number to a sharded history bitmap using the provided
    /// cursor.
    ///
    /// Uses the same shard-splitting logic as reth's `append_history_index`:
    /// - Seek the last shard (keyed with `u64::MAX`).
    /// - Append the new block number.
    /// - If the shard exceeds [`NUM_OF_INDICES_IN_SHARD`], split it.
    ///
    /// The caller must supply a reusable cursor to avoid creating a new one per
    /// key (which would be ~11 000 cursor allocations per Base mainnet block).
    fn append_history_index_with_cursor<T>(
        cursor: &mut (impl DbCursorRO<T> + DbCursorRW<T>),
        block_number: BlockNumber,
        sharded_key_factory: impl Fn(BlockNumber) -> T::Key,
    ) -> OpProofsStorageResult<()>
    where
        T: Table<Value = BlockNumberList>,
        T::Key: Clone,
    {
        let last_key = sharded_key_factory(u64::MAX);

        let mut last_shard = cursor
            .seek_exact(last_key.clone())?
            .map(|(_, list)| list)
            .unwrap_or_else(BlockNumberList::empty);

        last_shard
            .push(block_number)
            .map_err(|e| DatabaseError::Other(format!("IntegerList push error: {e}")))?;

        // Fast path: fits in one shard
        if last_shard.len() <= NUM_OF_INDICES_IN_SHARD as u64 {
            cursor.upsert(last_key, &last_shard)?;
            return Ok(());
        }

        // Slow path: rechunk
        // Delete the old u64::MAX shard first
        if cursor.seek_exact(last_key)?.is_some() {
            cursor.delete_current()?;
        }

        let all_values: Vec<u64> = last_shard.iter().collect();
        let chunks = all_values.chunks(NUM_OF_INDICES_IN_SHARD);
        let total_chunks = chunks.len();

        for (i, chunk) in all_values.chunks(NUM_OF_INDICES_IN_SHARD).enumerate() {
            let shard = BlockNumberList::new_pre_sorted(chunk.iter().copied());
            let key = if i < total_chunks - 1 {
                // Completed shard: use actual highest block number
                sharded_key_factory(*chunk.last().expect("non-empty chunk"))
            } else {
                // Last shard: sentinel
                sharded_key_factory(u64::MAX)
            };
            cursor.upsert(key, &shard)?;
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // History bitmap removal (for unwind / replace)
    // -------------------------------------------------------------------------

    /// Remove multiple block numbers from a single key's history bitmap shard(s).
    ///
    /// Reuses the provided cursor. Processes blocks in sorted order for
    /// sequential shard access. When a shard is found, ALL matching block
    /// numbers are removed in a single bitmap edit, so subsequent seeks for
    /// blocks in the same shard become no-ops.
    fn remove_blocks_from_history_shard<T>(
        cursor: &mut (impl DbCursorRO<T> + DbCursorRW<T>),
        blocks_to_remove: &BTreeSet<u64>,
        sharded_key_factory: impl Fn(u64) -> T::Key,
    ) -> OpProofsStorageResult<()>
    where
        T: Table<Value = BlockNumberList>,
        T::Key: Clone,
    {
        for &block_number in blocks_to_remove {
            let seek_key = sharded_key_factory(block_number);
            let Some((key, list)) = cursor.seek(seek_key)? else {
                continue;
            };

            if !list.contains(block_number) {
                // Already removed by a previous shard edit (batch removal),
                // or was never present.
                continue;
            }

            // Remove ALL target blocks from this shard in one pass.
            let filtered: Vec<u64> =
                list.iter().filter(|&bn| !blocks_to_remove.contains(&bn)).collect();

            if filtered.is_empty() {
                cursor.delete_current()?;
            } else {
                let new_list = BlockNumberList::new_pre_sorted(filtered.into_iter());
                cursor.upsert(key, &new_list)?;
            }
        }

        Ok(())
    }

    /// Prune-specific history removal: for a given logical key, seek its first
    /// history shard and walk forward, removing all block numbers that fall
    /// within `range`.  This is faster than [`Self::remove_blocks_from_history_shard`]
    /// for pruning because it requires only **one seek per unique key** (instead
    /// of one seek per block) and uses a simple range check instead of a
    /// set-membership lookup.
    fn prune_history_range_for_key<T>(
        cursor: &mut (impl DbCursorRO<T> + DbCursorRW<T>),
        range: &std::ops::RangeInclusive<u64>,
        first_shard_key: T::Key,
        same_logical_key: impl Fn(&T::Key) -> bool,
    ) -> OpProofsStorageResult<()>
    where
        T: Table<Value = BlockNumberList>,
        T::Key: Clone,
    {
        let mut entry = cursor.seek(first_shard_key)?;
        loop {
            let Some((key, list)) = entry else { break };

            if !same_logical_key(&key) {
                break;
            }

            let original_len = list.len() as usize;
            let filtered: Vec<u64> =
                list.iter().filter(|&bn| !range.contains(&bn)).collect();

            if filtered.is_empty() {
                // Entire shard pruned — delete and advance.
                cursor.delete_current()?;
                entry = cursor.current()?;
            } else if filtered.len() < original_len {
                // Partial prune — update shard and advance.
                let new_list = BlockNumberList::new_pre_sorted(filtered.into_iter());
                cursor.upsert(key, &new_list)?;
                entry = cursor.next()?;
            } else {
                // No blocks in this shard were in range.
                // If the shard's lowest block is past our range, stop.
                if list.iter().next().map_or(true, |first| first > *range.end()) {
                    break;
                }
                entry = cursor.next()?;
            }
        }

        Ok(())
    }

    /// Remove block numbers from all 4 history bitmap tables by reading the
    /// changeset tables to find exactly which keys were affected.
    ///
    /// Optimised path: each changeset table is walked **once**, entries are
    /// deduplicated by key into a `BTreeMap<key, BTreeSet<block_number>>`,
    /// and then each unique key's bitmap shard(s) are edited in a single
    /// batch operation through a **reused cursor**.
    fn remove_all_history_entries(
        &self,
        range: std::ops::RangeInclusive<u64>,
    ) -> OpProofsStorageResult<()> {
        // ---- Account trie history ----
        {
            let mut dedup: BTreeMap<StoredNibbles, BTreeSet<u64>> = BTreeMap::new();
            let mut cs_cursor = self.tx.cursor_read::<AccountTrieChangeSets>()?;
            let mut walker = cs_cursor.walk_range(range.clone())?;
            while let Some(Ok((block_number, entry))) = walker.next() {
                let nibbles = StoredNibbles(entry.nibbles.0);
                dedup.entry(nibbles).or_default().insert(block_number);
            }
            drop(walker);
            drop(cs_cursor);

            let mut hist_cursor = self.tx.cursor_write::<AccountsTrieHistory>()?;
            for (nibbles, blocks) in &dedup {
                Self::remove_blocks_from_history_shard(
                    &mut hist_cursor,
                    blocks,
                    |highest| AccountTrieShardedKey::new(nibbles.clone(), highest),
                )?;
            }
        }

        // ---- Storage trie history ----
        {
            let mut dedup: BTreeMap<(B256, StoredNibbles), BTreeSet<u64>> = BTreeMap::new();
            let mut cs_cursor = self.tx.cursor_read::<StorageTrieChangeSets>()?;
            let start = BlockNumberHashedAddress((*range.start(), B256::ZERO));
            let end = BlockNumberHashedAddress((*range.end(), B256::repeat_byte(0xff)));
            let mut walker = cs_cursor.walk_range(start..=end)?;
            while let Some(Ok((key, entry))) = walker.next() {
                let hashed_address = key.0 .1;
                let nibbles = StoredNibbles(entry.nibbles.0);
                dedup.entry((hashed_address, nibbles)).or_default().insert(key.0 .0);
            }
            drop(walker);
            drop(cs_cursor);

            let mut hist_cursor = self.tx.cursor_write::<StoragesTrieHistory>()?;
            for ((hashed_address, nibbles), blocks) in &dedup {
                Self::remove_blocks_from_history_shard(
                    &mut hist_cursor,
                    blocks,
                    |highest| {
                        StorageTrieShardedKey::new(*hashed_address, nibbles.clone(), highest)
                    },
                )?;
            }
        }

        // ---- Hashed accounts history ----
        {
            let mut dedup: BTreeMap<B256, BTreeSet<u64>> = BTreeMap::new();
            let mut cs_cursor = self.tx.cursor_read::<HashedAccountChangeSets>()?;
            let mut walker = cs_cursor.walk_range(range.clone())?;
            while let Some(Ok((block_number, entry))) = walker.next() {
                dedup.entry(entry.hashed_address).or_default().insert(block_number);
            }
            drop(walker);
            drop(cs_cursor);

            let mut hist_cursor = self.tx.cursor_write::<HashedAccountsHistory>()?;
            for (addr, blocks) in &dedup {
                Self::remove_blocks_from_history_shard(
                    &mut hist_cursor,
                    blocks,
                    |highest| HashedAccountShardedKey::new(*addr, highest),
                )?;
            }
        }

        // ---- Hashed storages history ----
        {
            let mut dedup: BTreeMap<(B256, B256), BTreeSet<u64>> = BTreeMap::new();
            let mut cs_cursor = self.tx.cursor_read::<HashedStorageChangeSets>()?;
            let start = BlockNumberHashedAddress((*range.start(), B256::ZERO));
            let end = BlockNumberHashedAddress((*range.end(), B256::repeat_byte(0xff)));
            let mut walker = cs_cursor.walk_range(start..=end)?;
            while let Some(Ok((key, entry))) = walker.next() {
                let hashed_address = key.0 .1;
                dedup.entry((hashed_address, entry.key)).or_default().insert(key.0 .0);
            }
            drop(walker);
            drop(cs_cursor);

            let mut hist_cursor = self.tx.cursor_write::<HashedStoragesHistory>()?;
            for ((hashed_address, storage_key), blocks) in &dedup {
                Self::remove_blocks_from_history_shard(
                    &mut hist_cursor,
                    blocks,
                    |highest| HashedStorageShardedKey {
                        hashed_address: *hashed_address,
                        sharded_key: ShardedKey::new(*storage_key, highest),
                    },
                )?;
            }
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Store trie updates for a single block
    // -------------------------------------------------------------------------

    /// Core write logic for a single block.
    ///
    /// 1. Read old values from current-state tables.
    /// 2. Write old values to changeset tables.
    /// 3. Append block number to history bitmap tables.
    /// 4. Write new values to current-state tables (or delete if removed).
    ///
    /// Returns the write counts.
    fn store_block_updates(
        &self,
        block_number: BlockNumber,
        block_state_diff: BlockStateDiff,
    ) -> OpProofsStorageResult<WriteCounts> {
        let BlockStateDiff { sorted_trie_updates, sorted_post_state } = block_state_diff;

        let mut counts = WriteCounts::default();

        // ---- Account Trie ----
        //
        // Open all three cursors (state, changeset, history) up front and
        // inline the bitmap append so we avoid collecting a Vec of keys.
        {
            let mut state_cursor = self.tx.cursor_write::<AccountsTrie>()?;
            let mut cs_cursor = self.tx.cursor_dup_write::<AccountTrieChangeSets>()?;
            let mut hist_cursor = self.tx.cursor_write::<AccountsTrieHistory>()?;

            for (nibbles, maybe_node) in sorted_trie_updates.account_nodes_ref() {
                let stored = StoredNibbles(nibbles.clone());

                // Read old value from current state (cursor positioned for reuse)
                let old_entry = state_cursor.seek_exact(stored.clone())?;
                let old_node = old_entry.as_ref().map(|(_, node)| node.clone());
                let had_old = old_entry.is_some();

                // Write old value to changeset
                let cs_entry = TrieChangeSetsEntry {
                    nibbles: StoredNibblesSubKey(nibbles.clone()),
                    node: old_node,
                };
                cs_cursor.append_dup(block_number, cs_entry)?;

                // Inline history bitmap append (avoids Vec + second loop)
                {
                    let key = stored.clone();
                    Self::append_history_index_with_cursor::<AccountsTrieHistory>(
                        &mut hist_cursor,
                        block_number,
                        |highest| AccountTrieShardedKey::new(key.clone(), highest),
                    )?;
                }

                // Update current state — reuse cursor position for deletions
                match maybe_node {
                    Some(node) => {
                        state_cursor.upsert(stored, node)?;
                    }
                    None => {
                        if had_old {
                            state_cursor.delete_current()?;
                        }
                    }
                }

                counts.account_trie_updates_written_total += 1;
            }
        }

        // ---- Storage Trie ----
        {
            let mut state_cursor = self.tx.cursor_dup_write::<StoragesTrie>()?;
            let mut cs_cursor = self.tx.cursor_dup_write::<StorageTrieChangeSets>()?;
            let mut hist_cursor = self.tx.cursor_write::<StoragesTrieHistory>()?;

            for (hashed_address, nodes) in sorted_trie_updates.storage_tries_ref() {
                let cs_key = BlockNumberHashedAddress((block_number, *hashed_address));

                if nodes.is_deleted {
                    // Wipe: iterate all existing storage trie nodes for this address
                    // and record them as deleted in changesets
                    if let Some((_key, first_entry)) =
                        state_cursor.seek_exact(*hashed_address)?
                    {
                        // Record first entry
                        let cs_entry = TrieChangeSetsEntry {
                            nibbles: first_entry.nibbles.clone(),
                            node: Some(first_entry.node.clone()),
                        };
                        cs_cursor.append_dup(cs_key.clone(), cs_entry)?;
                        Self::append_history_index_with_cursor::<StoragesTrieHistory>(
                            &mut hist_cursor,
                            block_number,
                            |highest| StorageTrieShardedKey::new(
                                *hashed_address,
                                StoredNibbles(first_entry.nibbles.0.clone()),
                                highest,
                            ),
                        )?;

                        // Record remaining entries
                        while let Some((_, entry)) = state_cursor.next_dup()? {
                            let cs_entry = TrieChangeSetsEntry {
                                nibbles: entry.nibbles.clone(),
                                node: Some(entry.node.clone()),
                            };
                            cs_cursor.append_dup(cs_key.clone(), cs_entry)?;
                            Self::append_history_index_with_cursor::<StoragesTrieHistory>(
                                &mut hist_cursor,
                                block_number,
                                |highest| StorageTrieShardedKey::new(
                                    *hashed_address,
                                    StoredNibbles(entry.nibbles.0.clone()),
                                    highest,
                                ),
                            )?;
                        }

                        // Delete all entries for this address
                        if state_cursor.seek_exact(*hashed_address)?.is_some() {
                            state_cursor.delete_current_duplicates()?;
                        }
                    }

                    counts.storage_trie_updates_written_total += 1;
                }

                for (nibbles, maybe_node) in nodes.storage_nodes_ref() {
                    let subkey = StoredNibblesSubKey(nibbles.clone());

                    // Read old value from current state (first seek positions
                    // the cursor — we reuse that position for the delete below
                    // to avoid a second seek_by_key_subkey).
                    let old_entry = state_cursor
                        .seek_by_key_subkey(*hashed_address, subkey.clone())?
                        .filter(|e| e.nibbles == subkey);
                    let had_old = old_entry.is_some();
                    let old_node = old_entry.map(|e| e.node);

                    // Write old value to changeset
                    let cs_entry = TrieChangeSetsEntry {
                        nibbles: subkey.clone(),
                        node: old_node,
                    };
                    cs_cursor.append_dup(cs_key.clone(), cs_entry)?;

                    // Inline history bitmap append (avoids clone + Vec)
                    {
                        let addr = *hashed_address;
                        let nib = StoredNibbles(nibbles.clone());
                        Self::append_history_index_with_cursor::<StoragesTrieHistory>(
                            &mut hist_cursor,
                            block_number,
                            |highest| StorageTrieShardedKey::new(addr, nib.clone(), highest),
                        )?;
                    }

                    // Update current state — the state_cursor position is
                    // preserved across the cs_cursor / hist_cursor operations
                    // above, so we can delete_current() without re-seeking.
                    match maybe_node {
                        Some(node) => {
                            if had_old {
                                state_cursor.delete_current()?;
                            }
                            state_cursor.upsert(
                                *hashed_address,
                                &StorageTrieEntry {
                                    nibbles: subkey,
                                    node: node.clone(),
                                },
                            )?;
                        }
                        None => {
                            if had_old {
                                state_cursor.delete_current()?;
                            }
                        }
                    }

                    counts.storage_trie_updates_written_total += 1;
                }
            }
        }

        // ---- Hashed Accounts ----
        {
            let mut state_cursor = self.tx.cursor_write::<HashedAccounts>()?;
            let mut cs_cursor = self.tx.cursor_dup_write::<HashedAccountChangeSets>()?;
            let mut hist_cursor = self.tx.cursor_write::<HashedAccountsHistory>()?;

            for (hashed_address, maybe_account) in sorted_post_state.accounts.iter() {
                // Read old value from current state (cursor positioned for reuse)
                let old_entry = state_cursor.seek_exact(*hashed_address)?;
                let old_account = old_entry.as_ref().map(|(_, acc)| *acc);
                let had_old = old_entry.is_some();

                // Write old value to changeset
                let cs_entry = HashedAccountBeforeTx::new(*hashed_address, old_account);
                cs_cursor.append_dup(block_number, cs_entry)?;

                // Inline history bitmap append
                {
                    let key = *hashed_address;
                    Self::append_history_index_with_cursor::<HashedAccountsHistory>(
                        &mut hist_cursor,
                        block_number,
                        |highest| HashedAccountShardedKey::new(key, highest),
                    )?;
                }

                // Update current state — reuse cursor position for deletions
                match maybe_account {
                    Some(account) => {
                        state_cursor.upsert(*hashed_address, account)?;
                    }
                    None => {
                        if had_old {
                            state_cursor.delete_current()?;
                        }
                    }
                }

                counts.hashed_accounts_written_total += 1;
            }
        }

        // ---- Hashed Storages ----
        {
            let mut state_cursor = self.tx.cursor_dup_write::<HashedStorages>()?;
            let mut cs_cursor = self.tx.cursor_dup_write::<HashedStorageChangeSets>()?;
            let mut hist_cursor = self.tx.cursor_write::<HashedStoragesHistory>()?;

            for (hashed_address, storage) in &sorted_post_state.storages {
                let cs_key = BlockNumberHashedAddress((block_number, *hashed_address));

                if storage.is_wiped() {
                    // Wipe: iterate all existing storage entries for this address,
                    // record them in changeset, and delete from current state
                    if let Some(entry) =
                        state_cursor.seek_by_key_subkey(*hashed_address, B256::ZERO)?
                    {
                        // Record first entry
                        cs_cursor.append_dup(
                            cs_key.clone(),
                            StorageEntry { key: entry.key, value: entry.value },
                        )?;
                        Self::append_history_index_with_cursor::<HashedStoragesHistory>(
                            &mut hist_cursor,
                            block_number,
                            |highest| HashedStorageShardedKey {
                                hashed_address: *hashed_address,
                                sharded_key: ShardedKey::new(entry.key, highest),
                            },
                        )?;

                        // Record remaining entries
                        while let Some(entry) = state_cursor.next_dup_val()? {
                            cs_cursor.append_dup(
                                cs_key.clone(),
                                StorageEntry { key: entry.key, value: entry.value },
                            )?;
                            Self::append_history_index_with_cursor::<HashedStoragesHistory>(
                                &mut hist_cursor,
                                block_number,
                                |highest| HashedStorageShardedKey {
                                    hashed_address: *hashed_address,
                                    sharded_key: ShardedKey::new(entry.key, highest),
                                },
                            )?;
                        }

                        // Delete all entries for this address
                        if state_cursor.seek_exact(*hashed_address)?.is_some() {
                            state_cursor.delete_current_duplicates()?;
                        }
                    }
                }

                for (storage_key, value) in storage.storage_slots_ref() {
                    // Read old value from current state (first seek positions
                    // the cursor — reused for delete below).
                    let old_entry = state_cursor
                        .seek_by_key_subkey(*hashed_address, *storage_key)?
                        .filter(|e| e.key == *storage_key);
                    let had_old = old_entry.is_some();
                    let old_value = old_entry.map(|e| e.value).unwrap_or(U256::ZERO);

                    // Write old value to changeset
                    let cs_entry = StorageEntry { key: *storage_key, value: old_value };
                    cs_cursor.append_dup(cs_key.clone(), cs_entry)?;

                    // Inline history bitmap append
                    {
                        let addr = *hashed_address;
                        let sk = *storage_key;
                        Self::append_history_index_with_cursor::<HashedStoragesHistory>(
                            &mut hist_cursor,
                            block_number,
                            |highest| HashedStorageShardedKey {
                                hashed_address: addr,
                                sharded_key: ShardedKey::new(sk, highest),
                            },
                        )?;
                    }

                    // Update current state — cursor position preserved, no
                    // re-seek needed.
                    if *value == U256::ZERO {
                        if had_old {
                            state_cursor.delete_current()?;
                        }
                    } else {
                        if had_old {
                            state_cursor.delete_current()?;
                        }
                        state_cursor.upsert(
                            *hashed_address,
                            &StorageEntry { key: *storage_key, value: *value },
                        )?;
                    }

                    counts.hashed_storages_written_total += 1;
                }
            }
        }

        Ok(counts)
    }

    /// Store updates for a single block with out-of-order validation.
    fn store_trie_updates_inner(
        &self,
        block_ref: BlockWithParent,
        block_state_diff: BlockStateDiff,
    ) -> OpProofsStorageResult<WriteCounts> {
        let block_number = block_ref.block.number;

        let latest_block_hash = match self.get_latest_block_number_hash_inner()? {
            Some((_num, hash)) => hash,
            None => B256::ZERO,
        };

        if latest_block_hash != block_ref.parent {
            return Err(OpProofsStorageError::OutOfOrder {
                block_number,
                parent_block_hash: block_ref.parent,
                latest_block_hash,
            });
        }

        let counts = self.store_block_updates(block_number, block_state_diff)?;

        self.set_latest_block_number_inner(block_number, block_ref.block.hash)?;

        Ok(counts)
    }

    // -------------------------------------------------------------------------
    // Prune helpers
    // -------------------------------------------------------------------------

    /// Delete changeset entries for a block range.
    fn prune_changesets(&self, range: std::ops::RangeInclusive<u64>) -> OpProofsStorageResult<()> {
        // Account trie changesets
        {
            let mut cursor = self.tx.cursor_write::<AccountTrieChangeSets>()?;
            let mut walker = cursor.walk_range(range.clone())?;
            while walker.next().is_some() {
                walker.delete_current()?;
            }
        }

        // Storage trie changesets
        {
            let mut cursor = self.tx.cursor_write::<StorageTrieChangeSets>()?;
            let start = BlockNumberHashedAddress((*range.start(), B256::ZERO));
            let end = BlockNumberHashedAddress((*range.end(), B256::repeat_byte(0xff)));
            let mut walker = cursor.walk_range(start..=end)?;
            while walker.next().is_some() {
                walker.delete_current()?;
            }
        }

        // Account changesets
        {
            let mut cursor = self.tx.cursor_write::<HashedAccountChangeSets>()?;
            let mut walker = cursor.walk_range(range.clone())?;
            while walker.next().is_some() {
                walker.delete_current()?;
            }
        }

        // Storage changesets
        {
            let mut cursor = self.tx.cursor_write::<HashedStorageChangeSets>()?;
            let start = BlockNumberHashedAddress((*range.start(), B256::ZERO));
            let end = BlockNumberHashedAddress((*range.end(), B256::repeat_byte(0xff)));
            let mut walker = cursor.walk_range(start..=end)?;
            while walker.next().is_some() {
                walker.delete_current()?;
            }
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Unwind helpers
    // -------------------------------------------------------------------------

    /// Unwind account trie: restore old values from changesets.
    fn unwind_account_trie(
        &self,
        range: std::ops::RangeInclusive<u64>,
    ) -> OpProofsStorageResult<()> {
        let mut cs_cursor = self.tx.cursor_read::<AccountTrieChangeSets>()?;
        let mut state_cursor = self.tx.cursor_write::<AccountsTrie>()?;

        // Walk changesets in REVERSE order (newest first) to restore correctly
        for block_number in (*range.start()..=*range.end()).rev() {
            let mut walker = cs_cursor.walk(Some(block_number))?;
            while let Some(Ok((bn, cs_entry))) = walker.next() {
                if bn != block_number {
                    break;
                }
                let path = StoredNibbles(cs_entry.nibbles.0);

                match cs_entry.node {
                    Some(old_node) => {
                        // Restore old node
                        state_cursor.upsert(path, &old_node)?;
                    }
                    None => {
                        // Node didn't exist before → delete it
                        if state_cursor.seek_exact(path)?.is_some() {
                            state_cursor.delete_current()?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Unwind storage trie: restore old values from changesets.
    fn unwind_storage_trie(
        &self,
        range: std::ops::RangeInclusive<u64>,
    ) -> OpProofsStorageResult<()> {
        let mut cs_cursor = self.tx.cursor_read::<StorageTrieChangeSets>()?;
        let mut state_cursor = self.tx.cursor_dup_write::<StoragesTrie>()?;

        for block_number in (*range.start()..=*range.end()).rev() {
            let start = BlockNumberHashedAddress((block_number, B256::ZERO));
            let end = BlockNumberHashedAddress((block_number, B256::repeat_byte(0xff)));
            let mut walker = cs_cursor.walk_range(start..=end)?;

            while let Some(Ok((key, cs_entry))) = walker.next() {
                let hashed_address = key.0 .1;
                let subkey = cs_entry.nibbles.clone();

                // Delete current entry if it exists
                if state_cursor
                    .seek_by_key_subkey(hashed_address, subkey.clone())?
                    .filter(|e| e.nibbles == subkey)
                    .is_some()
                {
                    state_cursor.delete_current()?;
                }

                // Restore old value if it existed
                if let Some(old_node) = cs_entry.node {
                    state_cursor.upsert(
                        hashed_address,
                        &StorageTrieEntry { nibbles: subkey, node: old_node },
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Unwind hashed accounts: restore old values from changesets.
    fn unwind_hashed_accounts(
        &self,
        range: std::ops::RangeInclusive<u64>,
    ) -> OpProofsStorageResult<()> {
        let mut cs_cursor = self.tx.cursor_read::<HashedAccountChangeSets>()?;
        let mut state_cursor = self.tx.cursor_write::<HashedAccounts>()?;

        for block_number in (*range.start()..=*range.end()).rev() {
            let mut walker = cs_cursor.walk(Some(block_number))?;
            while let Some(Ok((bn, cs_entry))) = walker.next() {
                if bn != block_number {
                    break;
                }

                match cs_entry.info {
                    Some(old_account) => {
                        state_cursor.upsert(cs_entry.hashed_address, &old_account)?;
                    }
                    None => {
                        // Account didn't exist before → delete
                        if state_cursor.seek_exact(cs_entry.hashed_address)?.is_some() {
                            state_cursor.delete_current()?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Unwind hashed storages: restore old values from changesets.
    fn unwind_hashed_storages(
        &self,
        range: std::ops::RangeInclusive<u64>,
    ) -> OpProofsStorageResult<()> {
        let mut cs_cursor = self.tx.cursor_read::<HashedStorageChangeSets>()?;
        let mut state_cursor = self.tx.cursor_dup_write::<HashedStorages>()?;

        for block_number in (*range.start()..=*range.end()).rev() {
            let start = BlockNumberHashedAddress((block_number, B256::ZERO));
            let end = BlockNumberHashedAddress((block_number, B256::repeat_byte(0xff)));
            let mut walker = cs_cursor.walk_range(start..=end)?;

            while let Some(Ok((key, cs_entry))) = walker.next() {
                let hashed_address = key.0 .1;

                // Delete current entry if it exists
                if state_cursor
                    .seek_by_key_subkey(hashed_address, cs_entry.key)?
                    .filter(|e| e.key == cs_entry.key)
                    .is_some()
                {
                    state_cursor.delete_current()?;
                }

                // Restore old value if it existed (non-zero)
                if cs_entry.value != U256::ZERO {
                    state_cursor.upsert(
                        hashed_address,
                        &StorageEntry { key: cs_entry.key, value: cs_entry.value },
                    )?;
                }
            }
        }

        Ok(())
    }
}

// =============================================================================
// OpProofsProviderRO
// =============================================================================

impl<TX: DbTx + Send + Sync + Debug + 'static> OpProofsProviderRO for MdbxProofsProviderV2<TX> {
    type StorageTrieCursor<'tx>
        = V2StorageTrieCursor<
        TX::DupCursor<StoragesTrie>,
        TX::Cursor<StoragesTrieHistory>,
        TX::DupCursor<StorageTrieChangeSets>,
    >
    where
        Self: 'tx,
        TX: 'tx;

    type AccountTrieCursor<'tx>
        = V2AccountTrieCursor<
        TX::Cursor<AccountsTrie>,
        TX::Cursor<AccountsTrieHistory>,
        TX::DupCursor<AccountTrieChangeSets>,
    >
    where
        Self: 'tx,
        TX: 'tx;

    type StorageCursor<'tx>
        = V2StorageCursor<
        TX::DupCursor<HashedStorages>,
        TX::Cursor<HashedStoragesHistory>,
        TX::DupCursor<HashedStorageChangeSets>,
    >
    where
        Self: 'tx,
        TX: 'tx;

    type AccountHashedCursor<'tx>
        = V2AccountCursor<
        TX::Cursor<HashedAccounts>,
        TX::Cursor<HashedAccountsHistory>,
        TX::DupCursor<HashedAccountChangeSets>,
    >
    where
        Self: 'tx,
        TX: 'tx;

    fn get_earliest_block_number(&self) -> OpProofsStorageResult<Option<(u64, B256)>> {
        self.get_block_number_hash_inner(ProofWindowKey::EarliestBlock)
    }

    fn get_latest_block_number(&self) -> OpProofsStorageResult<Option<(u64, B256)>> {
        let mut cursor = self.tx.cursor_read::<ProofWindow>()?;
        if let Some((_, val)) = cursor.seek_exact(ProofWindowKey::LatestBlock)? {
            return Ok(Some((val.number(), *val.hash())));
        }
        let earliest = cursor.seek_exact(ProofWindowKey::EarliestBlock)?;
        Ok(earliest.map(|(_, val)| (val.number(), *val.hash())))
    }

    fn storage_trie_cursor<'tx>(
        &self,
        hashed_address: B256,
        max_block_number: u64,
    ) -> OpProofsStorageResult<Self::StorageTrieCursor<'tx>> {
        let is_latest = self.is_latest_block(max_block_number)?;
        Ok(V2StorageTrieCursor::new(
            self.tx.cursor_dup_read::<StoragesTrie>()?,
            self.tx.cursor_read::<StoragesTrieHistory>()?,
            self.tx.cursor_read::<StoragesTrieHistory>()?,
            self.tx.cursor_dup_read::<StorageTrieChangeSets>()?,
            hashed_address,
            max_block_number,
            is_latest,
        ))
    }

    fn account_trie_cursor<'tx>(
        &self,
        max_block_number: u64,
    ) -> OpProofsStorageResult<Self::AccountTrieCursor<'tx>> {
        let is_latest = self.is_latest_block(max_block_number)?;
        Ok(V2AccountTrieCursor::new(
            self.tx.cursor_read::<AccountsTrie>()?,
            self.tx.cursor_read::<AccountsTrieHistory>()?,
            self.tx.cursor_read::<AccountsTrieHistory>()?,
            self.tx.cursor_dup_read::<AccountTrieChangeSets>()?,
            max_block_number,
            is_latest,
        ))
    }

    fn storage_hashed_cursor<'tx>(
        &self,
        hashed_address: B256,
        max_block_number: u64,
    ) -> OpProofsStorageResult<Self::StorageCursor<'tx>> {
        let is_latest = self.is_latest_block(max_block_number)?;
        Ok(V2StorageCursor::new(
            self.tx.cursor_dup_read::<HashedStorages>()?,
            self.tx.cursor_read::<HashedStoragesHistory>()?,
            self.tx.cursor_read::<HashedStoragesHistory>()?,
            self.tx.cursor_dup_read::<HashedStorageChangeSets>()?,
            hashed_address,
            max_block_number,
            is_latest,
        ))
    }

    fn account_hashed_cursor<'tx>(
        &self,
        max_block_number: u64,
    ) -> OpProofsStorageResult<Self::AccountHashedCursor<'tx>> {
        let is_latest = self.is_latest_block(max_block_number)?;
        Ok(V2AccountCursor::new(
            self.tx.cursor_read::<HashedAccounts>()?,
            self.tx.cursor_read::<HashedAccountsHistory>()?,
            self.tx.cursor_read::<HashedAccountsHistory>()?,
            self.tx.cursor_dup_read::<HashedAccountChangeSets>()?,
            max_block_number,
            is_latest,
        ))
    }

    fn fetch_trie_updates(&self, block_number: u64) -> OpProofsStorageResult<BlockStateDiff> {
        self.fetch_trie_updates_inner(block_number)
    }
}

// =============================================================================
// OpProofsProviderRw
// =============================================================================

impl<TX: DbTxMut + DbTx + Send + Sync + Debug + 'static> OpProofsProviderRw
    for MdbxProofsProviderV2<TX>
{
    fn store_trie_updates(
        &self,
        block_ref: BlockWithParent,
        block_state_diff: BlockStateDiff,
    ) -> OpProofsStorageResult<WriteCounts> {
        self.store_trie_updates_inner(block_ref, block_state_diff)
    }

    fn store_trie_updates_batch(
        &self,
        updates: Vec<(BlockWithParent, BlockStateDiff)>,
    ) -> OpProofsStorageResult<WriteCounts> {
        let mut total_counts = WriteCounts::default();
        for (block_ref, block_state_diff) in updates {
            let counts = self.store_trie_updates_inner(block_ref, block_state_diff)?;
            total_counts += counts;
        }
        Ok(total_counts)
    }

    fn prune_earliest_state(
        &self,
        new_earliest_block_ref: BlockWithParent,
    ) -> OpProofsStorageResult<WriteCounts> {
        let target_block = new_earliest_block_ref.block.number;

        let Some((earliest, _)) =
            self.get_block_number_hash_inner(ProofWindowKey::EarliestBlock)?
        else {
            return Ok(WriteCounts::default());
        };

        if earliest >= target_block {
            return Ok(WriteCounts::default());
        }

        let range = (earliest + 1)..=target_block;

        // ---- Single-pass per changeset table: DupSort-aware walk that
        //      collects unique keys, counts entries, and bulk-deletes each
        //      primary key via delete_current_duplicates() — all in one
        //      cursor traversal. ----

        let mut counts = WriteCounts::default();

        // Account trie  (Key = BlockNumber, SubKey = StoredNibblesSubKey)
        let acct_trie_keys = {
            let mut keys: BTreeSet<StoredNibbles> = BTreeSet::new();
            let mut cursor = self.tx.cursor_dup_write::<AccountTrieChangeSets>()?;
            let mut entry = cursor.seek(*range.start())?;
            while let Some((block_num, first_val)) = entry {
                if block_num > *range.end() { break; }
                counts.account_trie_updates_written_total += 1;
                keys.insert(StoredNibbles(first_val.nibbles.0));
                while let Some((_, val)) = cursor.next_dup()? {
                    counts.account_trie_updates_written_total += 1;
                    keys.insert(StoredNibbles(val.nibbles.0));
                }
                cursor.delete_current_duplicates()?;
                entry = cursor.current()?;
            }
            keys
        };

        // Storage trie  (Key = BlockNumberHashedAddress, SubKey = StoredNibblesSubKey)
        let stor_trie_keys = {
            let mut keys: BTreeSet<(B256, StoredNibbles)> = BTreeSet::new();
            let mut cursor = self.tx.cursor_dup_write::<StorageTrieChangeSets>()?;
            let start = BlockNumberHashedAddress((*range.start(), B256::ZERO));
            let end = BlockNumberHashedAddress((*range.end(), B256::repeat_byte(0xff)));
            let mut entry = cursor.seek(start)?;
            while let Some((key, first_val)) = entry {
                if key > end { break; }
                counts.storage_trie_updates_written_total += 1;
                keys.insert((key.0 .1, StoredNibbles(first_val.nibbles.0)));
                while let Some((k, val)) = cursor.next_dup()? {
                    counts.storage_trie_updates_written_total += 1;
                    keys.insert((k.0 .1, StoredNibbles(val.nibbles.0)));
                }
                cursor.delete_current_duplicates()?;
                entry = cursor.current()?;
            }
            keys
        };

        // Hashed accounts  (Key = BlockNumber, SubKey = B256)
        let acct_keys = {
            let mut keys: BTreeSet<B256> = BTreeSet::new();
            let mut cursor = self.tx.cursor_dup_write::<HashedAccountChangeSets>()?;
            let mut entry = cursor.seek(*range.start())?;
            while let Some((block_num, first_val)) = entry {
                if block_num > *range.end() { break; }
                counts.hashed_accounts_written_total += 1;
                keys.insert(first_val.hashed_address);
                while let Some((_, val)) = cursor.next_dup()? {
                    counts.hashed_accounts_written_total += 1;
                    keys.insert(val.hashed_address);
                }
                cursor.delete_current_duplicates()?;
                entry = cursor.current()?;
            }
            keys
        };

        // Hashed storages  (Key = BlockNumberHashedAddress, SubKey = B256)
        let stor_keys = {
            let mut keys: BTreeSet<(B256, B256)> = BTreeSet::new();
            let mut cursor = self.tx.cursor_dup_write::<HashedStorageChangeSets>()?;
            let start = BlockNumberHashedAddress((*range.start(), B256::ZERO));
            let end = BlockNumberHashedAddress((*range.end(), B256::repeat_byte(0xff)));
            let mut entry = cursor.seek(start)?;
            while let Some((key, first_val)) = entry {
                if key > end { break; }
                counts.hashed_storages_written_total += 1;
                keys.insert((key.0 .1, first_val.key));
                while let Some((k, val)) = cursor.next_dup()? {
                    counts.hashed_storages_written_total += 1;
                    keys.insert((k.0 .1, val.key));
                }
                cursor.delete_current_duplicates()?;
                entry = cursor.current()?;
            }
            keys
        };

        // ---- Phase B: history bitmap removal — 1 seek per unique key, range filter ----
        {
            let mut cursor = self.tx.cursor_write::<AccountsTrieHistory>()?;
            for nibbles in &acct_trie_keys {
                Self::prune_history_range_for_key(
                    &mut cursor,
                    &range,
                    AccountTrieShardedKey::new(nibbles.clone(), 0),
                    |k| k.key == *nibbles,
                )?;
            }
        }
        {
            let mut cursor = self.tx.cursor_write::<StoragesTrieHistory>()?;
            for (hashed_address, nibbles) in &stor_trie_keys {
                Self::prune_history_range_for_key(
                    &mut cursor,
                    &range,
                    StorageTrieShardedKey::new(*hashed_address, nibbles.clone(), 0),
                    |k| k.hashed_address == *hashed_address && k.key == *nibbles,
                )?;
            }
        }
        {
            let mut cursor = self.tx.cursor_write::<HashedAccountsHistory>()?;
            for addr in &acct_keys {
                Self::prune_history_range_for_key(
                    &mut cursor,
                    &range,
                    HashedAccountShardedKey::new(*addr, 0),
                    |k| k.0.key == *addr,
                )?;
            }
        }
        {
            let mut cursor = self.tx.cursor_write::<HashedStoragesHistory>()?;
            for (hashed_address, storage_key) in &stor_keys {
                Self::prune_history_range_for_key(
                    &mut cursor,
                    &range,
                    HashedStorageShardedKey {
                        hashed_address: *hashed_address,
                        sharded_key: ShardedKey::new(*storage_key, 0),
                    },
                    |k| {
                        k.hashed_address == *hashed_address
                            && k.sharded_key.key == *storage_key
                    },
                )?;
            }
        }

        // Changesets already deleted during the single-pass walk above.

        self.set_earliest_block_number_inner(target_block, new_earliest_block_ref.block.hash)?;

        Ok(counts)
    }

    fn unwind_history(&self, to: BlockWithParent) -> OpProofsStorageResult<()> {
        let Some((earliest, latest)) = self.get_proof_window_inner()? else {
            return Ok(());
        };

        if to.block.number > latest.number {
            return Ok(());
        }

        if to.block.number <= earliest.number {
            return Err(OpProofsStorageError::UnwindBeyondEarliest {
                unwind_block_number: to.block.number,
                earliest_block_number: earliest.number,
            });
        }

        let range = to.block.number..=latest.number;

        // Restore old values from changesets
        self.unwind_account_trie(range.clone())?;
        self.unwind_storage_trie(range.clone())?;
        self.unwind_hashed_accounts(range.clone())?;
        self.unwind_hashed_storages(range.clone())?;

        // Remove unwound block numbers from history bitmaps
        self.remove_all_history_entries(range.clone())?;

        // Delete changeset entries for unwound blocks
        self.prune_changesets(range)?;

        // Update latest block
        self.set_latest_block_number_inner(
            to.block.number.saturating_sub(1),
            to.parent,
        )?;

        Ok(())
    }

    fn replace_updates(
        &self,
        latest_common_block: BlockNumHash,
        mut blocks_to_add: Vec<(BlockWithParent, BlockStateDiff)>,
    ) -> OpProofsStorageResult<()> {
        blocks_to_add.sort_unstable_by_key(|(bwp, _)| bwp.block.number);

        if let Some((latest_number, _)) = self.get_latest_block_number_hash_inner()? {
            if latest_common_block.number < latest_number {
                let range = (latest_common_block.number + 1)..=latest_number;

                // Restore old values from changesets
                self.unwind_account_trie(range.clone())?;
                self.unwind_storage_trie(range.clone())?;
                self.unwind_hashed_accounts(range.clone())?;
                self.unwind_hashed_storages(range.clone())?;

                // Remove old block numbers from history bitmaps
                self.remove_all_history_entries(range.clone())?;

                // Delete changeset entries
                self.prune_changesets(range)?;
            }
        }

        self.set_latest_block_number_inner(
            latest_common_block.number,
            latest_common_block.hash,
        )?;

        for (block_ref, diff) in blocks_to_add {
            self.store_trie_updates_inner(block_ref, diff)?;
        }
        Ok(())
    }

    fn set_earliest_block_number(
        &self,
        block_number: u64,
        hash: B256,
    ) -> OpProofsStorageResult<()> {
        self.set_earliest_block_number_inner(block_number, hash)
    }

    fn commit(self) -> OpProofsStorageResult<()> {
        self.tx.commit()?;
        Ok(())
    }
}

// =============================================================================
// OpProofsInitProvider
// =============================================================================

impl<TX: DbTxMut + DbTx + Send + Sync + Debug + 'static> OpProofsInitProvider
    for MdbxProofsProviderV2<TX>
{
    fn initial_state_anchor(&self) -> OpProofsStorageResult<InitialStateAnchor> {
        let Some(block) = self.get_initial_state_anchor_inner()? else {
            return Ok(InitialStateAnchor::default());
        };

        let completed =
            self.get_block_number_hash_inner(ProofWindowKey::EarliestBlock)?.is_some();

        // Scan the last entry in each current-state table to determine resume
        // keys. This allows multi-step initialization: if the process is
        // interrupted, the next run picks up where it left off.
        let latest_hashed_account_key = self
            .tx
            .cursor_read::<HashedAccounts>()?
            .last()?
            .map(|(k, _)| k);

        let latest_hashed_storage_key = self
            .tx
            .cursor_read::<HashedStorages>()?
            .last()?
            .map(|(addr, entry)| HashedStorageKey::new(addr, entry.key));

        let latest_account_trie_key = self
            .tx
            .cursor_read::<AccountsTrie>()?
            .last()?
            .map(|(k, _)| k);

        let latest_storage_trie_key = self
            .tx
            .cursor_read::<StoragesTrie>()?
            .last()?
            .map(|(addr, entry)| StorageTrieKey::new(addr, StoredNibbles(entry.nibbles.0)));

        Ok(InitialStateAnchor {
            block: Some(block),
            status: if completed {
                InitialStateStatus::Completed
            } else {
                InitialStateStatus::InProgress
            },
            latest_account_trie_key,
            latest_storage_trie_key,
            latest_hashed_account_key,
            latest_hashed_storage_key,
        })
    }

    fn set_initial_state_anchor(&self, anchor: BlockNumHash) -> OpProofsStorageResult<()> {
        let mut cur = self.tx.cursor_write::<ProofWindow>()?;
        cur.insert(ProofWindowKey::InitialStateAnchor, &anchor.into())?;
        Ok(())
    }

    fn store_account_branches(
        &self,
        account_nodes: Vec<(Nibbles, Option<BranchNodeCompact>)>,
    ) -> OpProofsStorageResult<()> {
        if account_nodes.is_empty() {
            return Ok(());
        }

        let mut cursor = self.tx.cursor_write::<AccountsTrie>()?;
        for (nibbles, maybe_node) in account_nodes {
            if let Some(node) = maybe_node {
                cursor.append(StoredNibbles(nibbles), &node)?;
            }
        }
        Ok(())
    }

    fn store_storage_branches(
        &self,
        hashed_address: B256,
        storage_nodes: Vec<(Nibbles, Option<BranchNodeCompact>)>,
    ) -> OpProofsStorageResult<()> {
        if storage_nodes.is_empty() {
            return Ok(());
        }

        let mut cursor = self.tx.cursor_dup_write::<StoragesTrie>()?;
        for (nibbles, maybe_node) in storage_nodes {
            if let Some(node) = maybe_node {
                cursor.append_dup(
                    hashed_address,
                    StorageTrieEntry {
                        nibbles: StoredNibblesSubKey(nibbles),
                        node,
                    },
                )?;
            }
        }
        Ok(())
    }

    fn store_hashed_accounts(
        &self,
        accounts: Vec<(B256, Option<Account>)>,
    ) -> OpProofsStorageResult<()> {
        if accounts.is_empty() {
            return Ok(());
        }

        let mut cursor = self.tx.cursor_write::<HashedAccounts>()?;
        for (hashed_address, maybe_account) in accounts {
            if let Some(account) = maybe_account {
                cursor.append(hashed_address, &account)?;
            }
        }
        Ok(())
    }

    fn store_hashed_storages(
        &self,
        hashed_address: B256,
        storages: Vec<(B256, U256)>,
    ) -> OpProofsStorageResult<()> {
        if storages.is_empty() {
            return Ok(());
        }

        let mut cursor = self.tx.cursor_dup_write::<HashedStorages>()?;
        for (storage_key, value) in storages {
            cursor.append_dup(
                hashed_address,
                StorageEntry { key: storage_key, value },
            )?;
        }
        Ok(())
    }

    fn commit_initial_state(&self) -> OpProofsStorageResult<BlockNumHash> {
        let anchor = self
            .get_initial_state_anchor_inner()?
            .ok_or(OpProofsStorageError::NoBlocksFound)?;
        self.set_earliest_block_number_inner(anchor.number, anchor.hash)?;
        Ok(anchor)
    }

    fn commit(self) -> OpProofsStorageResult<()> {
        self.tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::models;
    use alloy_eips::NumHash;
    use reth_db::{
        cursor::DbDupCursorRO,
        mdbx::{init_db_for, DatabaseArguments},
        transaction::DbTx,
        Database, DatabaseEnv,
    };
    use reth_trie::{updates::{StorageTrieUpdates, TrieUpdates}, HashedStorage};
    use tempfile::TempDir;

    fn setup_db() -> DatabaseEnv {
        let tmp = TempDir::new().expect("create tmpdir");
        init_db_for::<_, models::Tables>(tmp, DatabaseArguments::default()).expect("init db")
    }

    fn make_block_ref(number: u64, hash: B256, parent: B256) -> BlockWithParent {
        BlockWithParent::new(parent, NumHash::new(number, hash))
    }

    fn sample_account() -> Account {
        Account { nonce: 1, balance: U256::from(100u64), ..Default::default() }
    }

    fn sample_node() -> BranchNodeCompact {
        BranchNodeCompact::new(0b1, 0, 0, vec![], Some(B256::repeat_byte(0xAB)))
    }

    // ========================== Init provider tests ==========================

    #[test]
    fn init_store_hashed_accounts_writes_to_current_state() {
        let db = setup_db();

        let addr = B256::from([0xAA; 32]);
        let account = sample_account();
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw tx"));
            provider.store_hashed_accounts(vec![(addr, Some(account))]).expect("write");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = tx.cursor_read::<HashedAccounts>().expect("cursor");
        let (k, v) = cur.seek_exact(addr).expect("seek").expect("exists");
        assert_eq!(k, addr);
        assert_eq!(v.nonce, account.nonce);
        assert_eq!(v.balance, account.balance);
    }

    #[test]
    fn init_store_hashed_storages_writes_to_current_state() {
        let db = setup_db();

        let addr = B256::from([0x11; 32]);
        let slot = B256::from([0x22; 32]);
        let val = U256::from(0x1234u64);
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw tx"));
            provider.store_hashed_storages(addr, vec![(slot, val)]).expect("write");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = tx.cursor_dup_read::<HashedStorages>().expect("cursor");
        let entry = cur.seek_by_key_subkey(addr, slot).expect("seek").expect("exists");
        assert_eq!(entry.key, slot);
        assert_eq!(entry.value, val);
    }

    #[test]
    fn init_store_account_branches_writes_to_current_state() {
        let db = setup_db();

        let path = Nibbles::from_nibbles_unchecked([0x12, 0x34]);
        let node = sample_node();
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw tx"));
            provider.store_account_branches(vec![(path, Some(node.clone()))]).expect("write");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = tx.cursor_read::<AccountsTrie>().expect("cursor");
        let (k, v) = cur.seek_exact(StoredNibbles(path)).expect("seek").expect("exists");
        assert_eq!(k.0, path);
        assert_eq!(v, node);
    }

    #[test]
    fn init_store_storage_branches_writes_to_current_state() {
        let db = setup_db();

        let addr = B256::from([0x55; 32]);
        let path = Nibbles::from_nibbles_unchecked([0x12, 0x34]);
        let node = sample_node();
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw tx"));
            provider.store_storage_branches(addr, vec![(path, Some(node.clone()))]).expect("write");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        let tx = db.tx().expect("ro tx");
        let mut cur = tx.cursor_dup_read::<StoragesTrie>().expect("cursor");
        let entry = cur
            .seek_by_key_subkey(addr, StoredNibblesSubKey(path))
            .expect("seek")
            .expect("exists");
        assert_eq!(entry.nibbles.0, path);
        assert_eq!(entry.node, node);
    }

    // ========================== Store + unwind tests ==========================

    #[test]
    fn store_and_read_trie_updates_account() {
        let db = setup_db();

        let addr = B256::from([0xAA; 32]);
        let initial_account = Account { nonce: 1, ..Default::default() };
        let updated_account = Account { nonce: 2, ..Default::default() };

        // Initialize state
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider.store_hashed_accounts(vec![(addr, Some(initial_account))]).expect("init");
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Store block 1 update
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            let mut post_state = HashedPostState::default();
            post_state.accounts.insert(addr, Some(updated_account));

            let diff = BlockStateDiff {
                sorted_trie_updates: TrieUpdates::default().into_sorted(),
                sorted_post_state: post_state.into_sorted(),
            };

            let block_ref = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
            provider.store_trie_updates(block_ref, diff).expect("store");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Verify current state has the updated account
        {
            let tx = db.tx().expect("ro");
            let mut cur = tx.cursor_read::<HashedAccounts>().expect("cursor");
            let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
            assert_eq!(acc.nonce, 2, "current state should have updated nonce");
        }

        // Verify changeset has the old account
        {
            let tx = db.tx().expect("ro");
            let mut cur = tx.cursor_dup_read::<HashedAccountChangeSets>().expect("cursor");
            let entry = cur.seek_by_key_subkey(1u64, addr).expect("seek").expect("exists");
            assert_eq!(entry.hashed_address, addr);
            assert_eq!(entry.info.unwrap().nonce, 1, "changeset should have old nonce");
        }
    }

    #[test]
    fn unwind_restores_old_state() {
        let db = setup_db();

        let addr = B256::from([0xAA; 32]);
        let account_v0 = Account { nonce: 0, ..Default::default() };
        let account_v1 = Account { nonce: 1, ..Default::default() };

        // Initialize with v0
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider.store_hashed_accounts(vec![(addr, Some(account_v0))]).expect("init");
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Store block 1
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            let mut post_state = HashedPostState::default();
            post_state.accounts.insert(addr, Some(account_v1));

            let diff = BlockStateDiff {
                sorted_trie_updates: TrieUpdates::default().into_sorted(),
                sorted_post_state: post_state.into_sorted(),
            };

            let block_ref = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
            provider.store_trie_updates(block_ref, diff).expect("store");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Verify v1 is current
        {
            let tx = db.tx().expect("ro");
            let mut cur = tx.cursor_read::<HashedAccounts>().expect("cursor");
            let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
            assert_eq!(acc.nonce, 1);
        }

        // Unwind block 1
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            let unwind_to = BlockWithParent::new(B256::ZERO, NumHash::new(1, B256::repeat_byte(0x01)));
            provider.unwind_history(unwind_to).expect("unwind");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Verify v0 is restored
        {
            let tx = db.tx().expect("ro");
            let mut cur = tx.cursor_read::<HashedAccounts>().expect("cursor");
            let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
            assert_eq!(acc.nonce, 0, "unwind should restore nonce to 0");
        }
    }

    #[test]
    fn store_creates_history_bitmap() {
        let db = setup_db();

        let addr = B256::from([0xBB; 32]);

        // Initialize
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider
                .store_hashed_accounts(vec![(addr, Some(Account::default()))])
                .expect("init");
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Store 3 blocks
        let mut parent_hash = B256::ZERO;
        for block_num in 1..=3 {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            let hash = B256::repeat_byte(block_num as u8);
            let mut post_state = HashedPostState::default();
            post_state.accounts.insert(
                addr,
                Some(Account { nonce: block_num, ..Default::default() }),
            );

            let diff = BlockStateDiff {
                sorted_trie_updates: TrieUpdates::default().into_sorted(),
                sorted_post_state: post_state.into_sorted(),
            };
            let block_ref = make_block_ref(block_num, hash, parent_hash);
            provider.store_trie_updates(block_ref, diff).expect("store");
            OpProofsProviderRw::commit(provider).expect("commit");
            parent_hash = hash;
        }

        // Verify history bitmap exists and contains blocks 1, 2, 3
        {
            let tx = db.tx().expect("ro");
            let mut cur = tx.cursor_read::<HashedAccountsHistory>().expect("cursor");
            let shard_key = HashedAccountShardedKey::new(addr, u64::MAX);
            let (_, bitmap) = cur.seek_exact(shard_key).expect("seek").expect("exists");
            let blocks: Vec<u64> = bitmap.iter().collect();
            assert_eq!(blocks, vec![1, 2, 3]);
        }
    }

    #[test]
    fn prune_removes_changesets() {
        let db = setup_db();

        let addr = B256::from([0xCC; 32]);

        // Initialize
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider
                .store_hashed_accounts(vec![(addr, Some(Account::default()))])
                .expect("init");
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Store blocks 1-3
        let mut parent_hash = B256::ZERO;
        for block_num in 1u64..=3 {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            let hash = B256::repeat_byte(block_num as u8);
            let mut post_state = HashedPostState::default();
            post_state.accounts.insert(
                addr,
                Some(Account { nonce: block_num, ..Default::default() }),
            );
            let diff = BlockStateDiff {
                sorted_trie_updates: TrieUpdates::default().into_sorted(),
                sorted_post_state: post_state.into_sorted(),
            };
            let block_ref = make_block_ref(block_num, hash, parent_hash);
            provider.store_trie_updates(block_ref, diff).expect("store");
            OpProofsProviderRw::commit(provider).expect("commit");
            parent_hash = hash;
        }

        // Prune blocks 1-2
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            let prune_ref = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));
            provider.prune_earliest_state(prune_ref).expect("prune");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Verify changesets for blocks 1 and 2 are gone
        {
            let tx = db.tx().expect("ro");
            let mut cur = tx.cursor_read::<HashedAccountChangeSets>().expect("cursor");
            // Block 1 should be gone
            assert!(cur.seek_exact(1u64).expect("seek").is_none(), "block 1 changeset should be pruned");
            // Block 2 should be gone
            assert!(cur.seek_exact(2u64).expect("seek").is_none(), "block 2 changeset should be pruned");
            // Block 3 should still exist
            assert!(cur.seek_exact(3u64).expect("seek").is_some(), "block 3 changeset should remain");
        }

        // Current state should still be at block 3
        {
            let tx = db.tx().expect("ro");
            let mut cur = tx.cursor_read::<HashedAccounts>().expect("cursor");
            let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
            assert_eq!(acc.nonce, 3, "current state should be at block 3");
        }
    }

    // ========================== Helpers ==========================

    /// Initialize database with accounts and set anchor at block 0.
    fn init_state(db: &DatabaseEnv, accounts: Vec<(B256, Option<Account>)>) {
        let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
        if !accounts.is_empty() {
            provider.store_hashed_accounts(accounts).expect("init accounts");
        }
        provider
            .set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO))
            .expect("anchor");
        provider.commit_initial_state().expect("commit init");
        OpProofsInitProvider::commit(provider).expect("commit");
    }

    /// Store a block diff.
    fn store_block(db: &DatabaseEnv, block_ref: BlockWithParent, diff: BlockStateDiff) {
        let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
        provider.store_trie_updates(block_ref, diff).expect("store");
        OpProofsProviderRw::commit(provider).expect("commit");
    }

    /// Create a diff with one account change.
    fn make_nonce_diff(addr: B256, nonce: u64) -> BlockStateDiff {
        let mut post_state = HashedPostState::default();
        post_state.accounts.insert(addr, Some(Account { nonce, ..Default::default() }));
        BlockStateDiff {
            sorted_trie_updates: TrieUpdates::default().into_sorted(),
            sorted_post_state: post_state.into_sorted(),
        }
    }

    // ========================== Store trie updates tests ==========================

    #[test]
    fn store_trie_updates_out_of_order_rejects() {
        let db = setup_db();
        init_state(&db, vec![]);

        // Store block 1
        let b1 = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
        store_block(&db, b1, BlockStateDiff::default());

        // Try to store block 2 with wrong parent
        let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
        let bad_block = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0xFF));
        let res = provider.store_trie_updates(bad_block, BlockStateDiff::default());
        assert!(matches!(res, Err(OpProofsStorageError::OutOfOrder { .. })));
    }

    #[test]
    fn store_trie_updates_comprehensive() {
        let db = setup_db();

        let addr1 = B256::from([0x11; 32]);
        let addr2 = B256::from([0x22; 32]);
        let slot1 = B256::from([0xA1; 32]);
        let path1 = Nibbles::from_nibbles_unchecked(vec![0, 1, 2, 3]);
        let path2 = Nibbles::from_nibbles_unchecked(vec![4, 5, 6, 7]);
        let removed_path = Nibbles::from_nibbles_unchecked(vec![7, 8, 9]);
        let storage_path1 = Nibbles::from_nibbles_unchecked(vec![1, 2, 3, 4]);

        let acc1_old = Account { nonce: 0, balance: U256::from(50), ..Default::default() };
        let acc1_new = Account { nonce: 1, balance: U256::from(100), ..Default::default() };
        let node1_old = BranchNodeCompact::new(0b1, 0, 0, vec![], Some(B256::repeat_byte(0x01)));
        let node1_new = BranchNodeCompact::default();
        let node2_new = BranchNodeCompact::default();
        let removed_node_old =
            BranchNodeCompact::new(0b1, 0, 0, vec![], Some(B256::repeat_byte(0x02)));
        let snode1_old = BranchNodeCompact::new(0b1, 0, 0, vec![], Some(B256::repeat_byte(0x03)));
        let snode1_new = BranchNodeCompact::default();
        let val1_old = U256::from(111u64);
        let val1_new = U256::from(1234u64);

        // Initialize state
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider
                .store_hashed_accounts(vec![(addr1, Some(acc1_old))])
                .expect("init accounts");
            provider.store_hashed_storages(addr1, vec![(slot1, val1_old)]).expect("init storage");
            provider
                .store_account_branches(vec![
                    (path1, Some(node1_old.clone())),
                    (removed_path, Some(removed_node_old.clone())),
                ])
                .expect("init account trie");
            provider
                .store_storage_branches(addr1, vec![(storage_path1, Some(snode1_old.clone()))])
                .expect("init storage trie");
            provider
                .set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO))
                .expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Build diff
        let mut trie_updates = TrieUpdates::default();
        trie_updates.account_nodes.insert(path1, node1_new.clone());
        trie_updates.account_nodes.insert(path2, node2_new.clone());
        trie_updates.removed_nodes.insert(removed_path);
        let mut stu1 = StorageTrieUpdates::default();
        stu1.storage_nodes.insert(storage_path1, snode1_new.clone());
        trie_updates.storage_tries.insert(addr1, stu1);

        let mut post_state = HashedPostState::default();
        post_state.accounts.insert(addr1, Some(acc1_new));
        post_state.accounts.insert(addr2, None); // deletion of non-existing
        let mut storage1 = HashedStorage::default();
        storage1.storage.insert(slot1, val1_new);
        post_state.storages.insert(addr1, storage1);

        let diff = BlockStateDiff {
            sorted_trie_updates: trie_updates.into_sorted(),
            sorted_post_state: post_state.into_sorted(),
        };

        let block = make_block_ref(42, B256::repeat_byte(0x42), B256::ZERO);
        store_block(&db, block, diff);

        // Verify current state
        let tx = db.tx().expect("ro");

        // Account: addr1 should have new account
        let mut acc_cur = tx.cursor_read::<HashedAccounts>().expect("cursor");
        let (_, acc) = acc_cur.seek_exact(addr1).expect("seek").expect("exists");
        assert_eq!(acc.nonce, acc1_new.nonce);
        assert!(acc_cur.seek_exact(addr2).expect("seek").is_none(), "addr2 was never created");

        // Storage: addr1/slot1 should have new value
        let mut stor_cur = tx.cursor_dup_read::<HashedStorages>().expect("cursor");
        let entry = stor_cur.seek_by_key_subkey(addr1, slot1).expect("seek").expect("exists");
        assert_eq!(entry.value, val1_new);

        // Account trie: path1 new, path2 new, removed_path gone
        let mut trie_cur = tx.cursor_read::<AccountsTrie>().expect("cursor");
        let (_, n) = trie_cur.seek_exact(StoredNibbles(path1)).expect("seek").expect("exists");
        assert_eq!(n, node1_new);
        let (_, n2) = trie_cur.seek_exact(StoredNibbles(path2)).expect("seek").expect("exists");
        assert_eq!(n2, node2_new);
        assert!(
            trie_cur.seek_exact(StoredNibbles(removed_path)).expect("seek").is_none(),
            "removed path should be gone"
        );

        // Storage trie: addr1/storage_path1 should have new node
        let mut strie_cur = tx.cursor_dup_read::<StoragesTrie>().expect("cursor");
        let e = strie_cur
            .seek_by_key_subkey(addr1, StoredNibblesSubKey(storage_path1))
            .expect("seek")
            .expect("exists");
        assert_eq!(e.node, snode1_new);

        // Verify account changeset has old values
        let mut cs_cur = tx.cursor_read::<HashedAccountChangeSets>().expect("cursor");
        let mut entries = Vec::new();
        let mut walker = cs_cur.walk(Some(42u64)).expect("walk");
        while let Some(Ok((bn, entry))) = walker.next() {
            if bn != 42 {
                break;
            }
            entries.push(entry);
        }
        assert!(entries.iter().any(|e| e.hashed_address == addr1 && e.info == Some(acc1_old)));

        // Verify account trie changeset has old values
        let mut tcs_cur = tx.cursor_read::<AccountTrieChangeSets>().expect("cursor");
        let mut tentries = Vec::new();
        let mut walker = tcs_cur.walk(Some(42u64)).expect("walk");
        while let Some(Ok((bn, entry))) = walker.next() {
            if bn != 42 {
                break;
            }
            tentries.push(entry);
        }
        assert!(tentries
            .iter()
            .any(|e| e.nibbles.0 == path1 && e.node == Some(node1_old.clone())));
        assert!(tentries
            .iter()
            .any(|e| e.nibbles.0 == removed_path && e.node == Some(removed_node_old.clone())));
        assert!(tentries.iter().any(|e| e.nibbles.0 == path2 && e.node.is_none()));

        // Verify ProofWindow latest
        let mut pw_cur = tx.cursor_read::<ProofWindow>().expect("cursor");
        let (_, val) =
            pw_cur.seek_exact(ProofWindowKey::LatestBlock).expect("seek").expect("exists");
        assert_eq!(val.number(), 42);
    }

    #[test]
    fn store_trie_updates_empty_collections() {
        let db = setup_db();
        init_state(&db, vec![]);

        let block = make_block_ref(42, B256::repeat_byte(0x42), B256::ZERO);
        store_block(&db, block, BlockStateDiff::default());

        // All changeset tables should be empty
        let tx = db.tx().expect("ro");
        let mut cur1 = tx.cursor_read::<HashedAccountChangeSets>().expect("cursor");
        assert!(cur1.first().expect("first").is_none(), "Account changesets should be empty");
        let mut cur2 = tx.cursor_read::<AccountTrieChangeSets>().expect("cursor");
        assert!(cur2.first().expect("first").is_none(), "Account trie changesets should be empty");

        // ProofWindow should be updated
        let mut pw_cur = tx.cursor_read::<ProofWindow>().expect("cursor");
        let (_, val) =
            pw_cur.seek_exact(ProofWindowKey::LatestBlock).expect("seek").expect("exists");
        assert_eq!(val.number(), 42);
    }

    #[test]
    fn store_trie_updates_multiple_blocks() {
        let db = setup_db();
        let addr = B256::from([0x21; 32]);
        init_state(&db, vec![(addr, Some(Account::default()))]);

        let b1 = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
        store_block(&db, b1, make_nonce_diff(addr, 10));

        let b2 = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));
        store_block(&db, b2, make_nonce_diff(addr, 20));

        // Current state should have latest nonce
        let tx = db.tx().expect("ro");
        let mut cur = tx.cursor_read::<HashedAccounts>().expect("cursor");
        let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
        assert_eq!(acc.nonce, 20);

        // Changeset at block 1 should have old nonce (0)
        let mut cs = tx.cursor_read::<HashedAccountChangeSets>().expect("cursor");
        let (_, entry) = cs.seek_exact(1u64).expect("seek").expect("exists");
        assert_eq!(entry.info.unwrap().nonce, 0);

        // Changeset at block 2 should have old nonce (10)
        let (_, entry2) = cs.seek_exact(2u64).expect("seek").expect("exists");
        assert_eq!(entry2.info.unwrap().nonce, 10);
    }

    #[test]
    fn store_trie_updates_deleted_account_trie() {
        let db = setup_db();

        let acc_path = Nibbles::from_nibbles_unchecked([0x0A, 0x0B, 0x0C]);
        let node = sample_node();

        // Initialize with a trie node
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider.store_account_branches(vec![(acc_path, Some(node.clone()))]).expect("init");
            provider
                .set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO))
                .expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Store block that removes the node
        let mut trie_updates = TrieUpdates::default();
        trie_updates.removed_nodes.insert(acc_path);
        let diff = BlockStateDiff {
            sorted_trie_updates: trie_updates.into_sorted(),
            ..Default::default()
        };
        let block = make_block_ref(7, B256::repeat_byte(0x07), B256::ZERO);
        store_block(&db, block, diff);

        // Current state should not have the node
        let tx = db.tx().expect("ro");
        let mut cur = tx.cursor_read::<AccountsTrie>().expect("cursor");
        assert!(
            cur.seek_exact(StoredNibbles(acc_path)).expect("seek").is_none(),
            "node should be removed from current state"
        );

        // Changeset should have the old node
        let mut cs = tx.cursor_read::<AccountTrieChangeSets>().expect("cursor");
        let (_, entry) = cs.seek_exact(7u64).expect("seek").expect("exists");
        assert_eq!(entry.node, Some(node), "changeset should have old node");
    }

    #[test]
    fn store_trie_updates_deleted_storage_trie() {
        let db = setup_db();

        let addr = B256::from([0xAB; 32]);
        let st_path = Nibbles::from_nibbles_unchecked([0x01, 0x02, 0x03]);
        let node = sample_node();

        // Initialize with a storage trie node
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider
                .store_storage_branches(addr, vec![(st_path, Some(node.clone()))])
                .expect("init");
            provider
                .set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO))
                .expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Store block that removes the storage trie node
        let mut trie_updates = TrieUpdates::default();
        let mut st_updates = StorageTrieUpdates::default();
        st_updates.removed_nodes.insert(st_path);
        trie_updates.storage_tries.insert(addr, st_updates);
        let diff = BlockStateDiff {
            sorted_trie_updates: trie_updates.into_sorted(),
            ..Default::default()
        };
        let block = make_block_ref(8, B256::repeat_byte(0x08), B256::ZERO);
        store_block(&db, block, diff);

        // Current state should not have the node
        let tx = db.tx().expect("ro");
        let mut cur = tx.cursor_dup_read::<StoragesTrie>().expect("cursor");
        let result = cur
            .seek_by_key_subkey(addr, StoredNibblesSubKey(st_path))
            .expect("seek")
            .filter(|e| e.nibbles.0 == st_path);
        assert!(result.is_none(), "node should be removed from current state");

        // Changeset should have the old node
        let mut cs = tx.cursor_read::<StorageTrieChangeSets>().expect("cursor");
        let start = BlockNumberHashedAddress((8, B256::ZERO));
        let (_, entry) = cs.seek(start).expect("seek").expect("exists");
        assert_eq!(entry.node, Some(node), "changeset should have old node");
    }

    #[test]
    fn store_trie_updates_wiped_storage_trie_nodes() {
        let db = setup_db();

        let addr_wiped = B256::from([0x10; 32]);
        let addr_live = B256::from([0xF0; 32]);
        let p1 = Nibbles::from_nibbles_unchecked([0x01, 0x02]);
        let p2 = Nibbles::from_nibbles_unchecked([0x0A, 0x0B, 0x0C]);
        let n1 = BranchNodeCompact::default();
        let n2 = BranchNodeCompact::default();

        // Seed storage trie nodes for addr_wiped
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider
                .store_storage_branches(
                    addr_wiped,
                    vec![(p1, Some(n1.clone())), (p2, Some(n2.clone()))],
                )
                .expect("seed");
            provider
                .set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO))
                .expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Build diff that wipes addr_wiped's storage trie and adds a node for addr_live
        let mut trie_updates = TrieUpdates::default();
        let mut wiped_updates = StorageTrieUpdates::default();
        wiped_updates.set_deleted(true);
        trie_updates.storage_tries.insert(addr_wiped, wiped_updates);

        let live_path = Nibbles::from_nibbles_unchecked([0xEE, 0xFF]);
        let live_node = BranchNodeCompact::default();
        let mut live_updates = StorageTrieUpdates::default();
        live_updates.storage_nodes.insert(live_path, live_node.clone());
        trie_updates.storage_tries.insert(addr_live, live_updates);

        let diff = BlockStateDiff {
            sorted_trie_updates: trie_updates.into_sorted(),
            ..Default::default()
        };
        let block = make_block_ref(123, B256::repeat_byte(0x7B), B256::ZERO);
        store_block(&db, block, diff);

        // Verify: addr_wiped's storage trie nodes should be deleted from current state
        let tx = db.tx().expect("ro");
        let mut cur = tx.cursor_dup_read::<StoragesTrie>().expect("cursor");
        assert!(
            cur.seek_exact(addr_wiped).expect("seek").is_none(),
            "wiped address should have no storage trie nodes"
        );

        // addr_live should have its node
        let e = cur
            .seek_by_key_subkey(addr_live, StoredNibblesSubKey(live_path))
            .expect("seek")
            .expect("exists");
        assert_eq!(e.node, live_node);
    }

    #[test]
    fn store_trie_updates_wiped_storage() {
        let db = setup_db();

        let addr = B256::from([0x55; 32]);
        let s1 = B256::from([0x01; 32]);
        let s2 = B256::from([0x02; 32]);
        let v1 = U256::from(111u64);
        let v2 = U256::from(222u64);

        // Seed prior storage
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider.store_hashed_storages(addr, vec![(s1, v1), (s2, v2)]).expect("seed");
            provider
                .set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO))
                .expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Build diff that wipes storage
        let mut post_state = HashedPostState::default();
        post_state.storages.insert(addr, HashedStorage::new(true));
        let diff = BlockStateDiff {
            sorted_trie_updates: TrieUpdates::default().into_sorted(),
            sorted_post_state: post_state.into_sorted(),
        };
        let block = make_block_ref(42, B256::repeat_byte(0x42), B256::ZERO);
        store_block(&db, block, diff);

        // Current state: slots should be deleted
        let tx = db.tx().expect("ro");
        let mut cur = tx.cursor_dup_read::<HashedStorages>().expect("cursor");
        assert!(
            cur.seek_exact(addr).expect("seek").is_none(),
            "wiped storage should have no entries in current state"
        );

        // Changeset should have old values
        let mut cs = tx.cursor_read::<HashedStorageChangeSets>().expect("cursor");
        let start = BlockNumberHashedAddress((42, addr));
        let mut old_values = Vec::new();
        let mut walker = cs.walk(Some(start)).expect("walk");
        while let Some(Ok((key, entry))) = walker.next() {
            if key.0 .0 != 42 || key.0 .1 != addr {
                break;
            }
            old_values.push((entry.key, entry.value));
        }
        assert!(old_values.iter().any(|(k, v)| *k == s1 && *v == v1));
        assert!(old_values.iter().any(|(k, v)| *k == s2 && *v == v2));
    }

    #[test]
    fn store_trie_updates_wiped_and_non_wiped_mixed_order() {
        let db = setup_db();

        let addr_wiped = B256::from([0x01; 32]);
        let addr_live = B256::from([0xF0; 32]);
        let ws1 = B256::from([0xA1; 32]);
        let wv1 = U256::from(111u64);
        let ls1 = B256::from([0xB1; 32]);
        let lv1_old = U256::from(333u64);
        let lv1_new = U256::from(999u64);

        // Seed storage for both addresses
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider.store_hashed_storages(addr_wiped, vec![(ws1, wv1)]).expect("seed wiped");
            provider.store_hashed_storages(addr_live, vec![(ls1, lv1_old)]).expect("seed live");
            provider
                .set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO))
                .expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Build diff: wipe addr_wiped, update addr_live
        let mut post_state = HashedPostState::default();
        post_state.storages.insert(addr_wiped, HashedStorage::new(true));
        let mut live_storage = HashedStorage::default();
        live_storage.storage.insert(ls1, lv1_new);
        post_state.storages.insert(addr_live, live_storage);

        let diff = BlockStateDiff {
            sorted_trie_updates: TrieUpdates::default().into_sorted(),
            sorted_post_state: post_state.into_sorted(),
        };
        let block = make_block_ref(77, B256::repeat_byte(0x4D), B256::ZERO);
        store_block(&db, block, diff);

        // Verify: wiped address has no storage
        let tx = db.tx().expect("ro");
        let mut cur = tx.cursor_dup_read::<HashedStorages>().expect("cursor");
        assert!(
            cur.seek_exact(addr_wiped).expect("seek").is_none(),
            "wiped addr should have no storage"
        );

        // Live address has new value
        let entry = cur.seek_by_key_subkey(addr_live, ls1).expect("seek").expect("exists");
        assert_eq!(entry.value, lv1_new);
    }

    // ========================== Fetch tests ==========================

    #[test]
    fn fetch_trie_updates_basic() {
        let db = setup_db();

        let addr1 = B256::from([0x11; 32]);
        let addr2 = B256::from([0x22; 32]);
        let slot1 = B256::from([0xA1; 32]);
        let path1 = Nibbles::from_nibbles_unchecked(vec![0, 1, 2, 3]);
        let acc1_old = Account { nonce: 0, ..Default::default() };
        let acc1_new = Account { nonce: 1, balance: U256::from(100), ..Default::default() };
        let node1_old = sample_node();
        let node1_new = BranchNodeCompact::default();
        let val1_old = U256::from(111u64);
        let val1_new = U256::from(1234u64);

        // Initialize
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider
                .store_hashed_accounts(vec![(addr1, Some(acc1_old))])
                .expect("init accounts");
            provider.store_hashed_storages(addr1, vec![(slot1, val1_old)]).expect("init storage");
            provider
                .store_account_branches(vec![(path1, Some(node1_old))])
                .expect("init trie");
            provider
                .set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO))
                .expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Build diff and store
        let mut trie_updates = TrieUpdates::default();
        trie_updates.account_nodes.insert(path1, node1_new.clone());

        let mut post_state = HashedPostState::default();
        post_state.accounts.insert(addr1, Some(acc1_new));
        post_state.accounts.insert(addr2, None); // deletion
        let mut storage1 = HashedStorage::default();
        storage1.storage.insert(slot1, val1_new);
        post_state.storages.insert(addr1, storage1);

        let diff = BlockStateDiff {
            sorted_trie_updates: trie_updates.into_sorted(),
            sorted_post_state: post_state.into_sorted(),
        };
        let block = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
        store_block(&db, block, diff);

        // Fetch block 1
        let provider = MdbxProofsProviderV2::new(db.tx().expect("ro"));
        let got = provider.fetch_trie_updates(1).expect("fetch");

        // Verify: accounts should have current values
        assert!(got.sorted_post_state.accounts.iter().any(|(a, v)| *a == addr1 && v == &Some(acc1_new)));

        // Verify: trie updates should have current node
        assert!(!got.sorted_trie_updates.account_nodes_ref().is_empty());

        // Verify: storages should have current value
        assert!(!got.sorted_post_state.storages.is_empty());
    }

    #[test]
    fn fetch_trie_updates_empty_changeset() {
        let db = setup_db();
        init_state(&db, vec![]);

        let block = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
        store_block(&db, block, BlockStateDiff::default());

        let provider = MdbxProofsProviderV2::new(db.tx().expect("ro"));
        let got = provider.fetch_trie_updates(1).expect("fetch");
        assert!(got.sorted_trie_updates.account_nodes_ref().is_empty());
        assert!(got.sorted_trie_updates.storage_tries_ref().is_empty());
        assert!(got.sorted_post_state.accounts.is_empty());
        assert!(got.sorted_post_state.storages.is_empty());
    }

    // ========================== Proof window tests ==========================

    #[test]
    fn test_proof_window() {
        let db = setup_db();

        // Initial state: no values set
        {
            let provider = MdbxProofsProviderV2::new(db.tx().expect("ro"));
            assert_eq!(provider.get_earliest_block_number().expect("get"), None);
        }

        // Set earliest
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider.set_earliest_block_number(42, B256::repeat_byte(0x42)).expect("set");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Verify
        {
            let provider = MdbxProofsProviderV2::new(db.tx().expect("ro"));
            let earliest = provider.get_earliest_block_number().expect("get");
            assert_eq!(earliest, Some((42, B256::repeat_byte(0x42))));

            // Latest should fall back to earliest when not set
            let latest = provider.get_latest_block_number().expect("get");
            assert_eq!(latest, Some((42, B256::repeat_byte(0x42))));
        }

        // Update earliest
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider.set_earliest_block_number(100, B256::repeat_byte(0x64)).expect("set");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Verify update
        {
            let provider = MdbxProofsProviderV2::new(db.tx().expect("ro"));
            let earliest = provider.get_earliest_block_number().expect("get");
            assert_eq!(earliest, Some((100, B256::repeat_byte(0x64))));
        }
    }

    // ========================== Prune tests ==========================

    #[test]
    fn test_prune_earliest_state_no_op() {
        let db = setup_db();
        init_state(&db, vec![]);

        // Attempt to prune with a block that is not newer than earliest
        let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
        let block_0 = make_block_ref(0, B256::repeat_byte(0x00), B256::ZERO);
        let counts = provider.prune_earliest_state(block_0).expect("prune");
        assert_eq!(counts, WriteCounts::default());
    }

    #[test]
    fn test_prune_earliest_state_uninitialized_guard() {
        let db = setup_db();
        // Don't initialize — earliest is None

        let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
        let target = make_block_ref(5, B256::repeat_byte(0x05), B256::ZERO);
        let counts = provider.prune_earliest_state(target).expect("prune");
        assert_eq!(counts, WriteCounts::default());
    }

    #[test]
    fn test_prune_earliest_state_overlapping_keys() {
        let db = setup_db();

        let addr = B256::from([0xDD; 32]);
        let acc1 = Account { nonce: 1, ..Default::default() };
        let acc2 = Account { nonce: 2, ..Default::default() };

        init_state(&db, vec![(addr, Some(Account::default()))]);

        // Block 1: update
        let b1 = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
        store_block(&db, b1, make_nonce_diff(addr, acc1.nonce));

        // Block 2: update same key
        let b2 = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));
        store_block(&db, b2, make_nonce_diff(addr, acc2.nonce));

        // Prune to block 3
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            let b3 = make_block_ref(3, B256::repeat_byte(0x03), B256::repeat_byte(0x02));
            provider.prune_earliest_state(b3).expect("prune");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Current state should still have nonce 2 (latest)
        let tx = db.tx().expect("ro");
        let mut cur = tx.cursor_read::<HashedAccounts>().expect("cursor");
        let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
        assert_eq!(acc.nonce, 2, "current state should still have latest value");

        // Changesets for blocks 1 and 2 should be gone
        let mut cs = tx.cursor_read::<HashedAccountChangeSets>().expect("cursor");
        assert!(cs.seek_exact(1u64).expect("seek").is_none());
        assert!(cs.seek_exact(2u64).expect("seek").is_none());
    }

    #[test]
    fn test_prune_earliest_state_comprehensive() {
        let db = setup_db();

        let addr = B256::from([0xEE; 32]);
        let slot = B256::from([0xAA; 32]);
        let path = Nibbles::from_nibbles_unchecked([0x01]);
        let storage_path = Nibbles::from_nibbles_unchecked([0x03]);
        let node_old = sample_node();
        let snode_old = sample_node();

        // Initialize with all 4 data types
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider
                .store_hashed_accounts(vec![(addr, Some(Account::default()))])
                .expect("init");
            provider
                .store_hashed_storages(addr, vec![(slot, U256::from(100u64))])
                .expect("init");
            provider
                .store_account_branches(vec![(path, Some(node_old.clone()))])
                .expect("init");
            provider
                .store_storage_branches(addr, vec![(storage_path, Some(snode_old.clone()))])
                .expect("init");
            provider
                .set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO))
                .expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Block 1: update account and trie
        let b1 = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
        {
            let mut trie_updates = TrieUpdates::default();
            trie_updates.account_nodes.insert(path, BranchNodeCompact::default());
            let mut post_state = HashedPostState::default();
            post_state.accounts.insert(addr, Some(Account { nonce: 1, ..Default::default() }));
            let mut storage = HashedStorage::default();
            storage.storage.insert(slot, U256::from(200u64));
            post_state.storages.insert(addr, storage);
            let diff = BlockStateDiff {
                sorted_trie_updates: trie_updates.into_sorted(),
                sorted_post_state: post_state.into_sorted(),
            };
            store_block(&db, b1, diff);
        }

        // Block 2: update account again
        let b2 = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));
        store_block(&db, b2, make_nonce_diff(addr, 2));

        // Prune to block 3
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            let b3 = make_block_ref(3, B256::repeat_byte(0x03), B256::repeat_byte(0x02));
            provider.prune_earliest_state(b3).expect("prune");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Current state should still have latest values
        let tx = db.tx().expect("ro");
        let mut acc_cur = tx.cursor_read::<HashedAccounts>().expect("cursor");
        let (_, acc) = acc_cur.seek_exact(addr).expect("seek").expect("exists");
        assert_eq!(acc.nonce, 2);

        // Changesets should be gone for blocks 1 and 2
        let mut cs = tx.cursor_read::<HashedAccountChangeSets>().expect("cursor");
        assert!(cs.seek_exact(1u64).expect("seek").is_none());
        assert!(cs.seek_exact(2u64).expect("seek").is_none());

        let mut tcs = tx.cursor_read::<AccountTrieChangeSets>().expect("cursor");
        assert!(tcs.seek_exact(1u64).expect("seek").is_none());
    }

    #[test]
    fn test_prune_earliest_state_returns_correct_counts() {
        let db = setup_db();
        let addr = B256::from([0xFF; 32]);
        init_state(&db, vec![(addr, Some(Account::default()))]);

        // Block 1: update
        let b1 = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
        store_block(&db, b1, make_nonce_diff(addr, 1));

        // Block 2: update
        let b2 = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));
        store_block(&db, b2, make_nonce_diff(addr, 2));

        // Prune to block 2 — should remove changeset for block 1
        let counts = {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            let prune_ref =
                make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));
            let c = provider.prune_earliest_state(prune_ref).expect("prune");
            OpProofsProviderRw::commit(provider).expect("commit");
            c
        };

        // Range is (earliest+1)..=target = 1..=2, pruning changesets for both blocks
        assert_eq!(counts.hashed_accounts_written_total, 2);
    }

    // ========================== Replace tests ==========================

    #[test]
    fn replace_updates_prunes_and_adds_new_chain() {
        let db = setup_db();
        let addr = B256::from([0xAB; 32]);
        init_state(&db, vec![(addr, Some(Account::default()))]);

        // Build initial chain: 1 -> 2 -> 3
        let b1 = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
        let b2 = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));
        let b3 = make_block_ref(3, B256::repeat_byte(0x03), B256::repeat_byte(0x02));

        store_block(&db, b1, make_nonce_diff(addr, 10));
        store_block(&db, b2, make_nonce_diff(addr, 20));
        store_block(&db, b3, make_nonce_diff(addr, 30));

        // Sanity: current state has nonce 30
        {
            let tx = db.tx().expect("ro");
            let mut cur = tx.cursor_read::<HashedAccounts>().expect("cursor");
            let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
            assert_eq!(acc.nonce, 30);
        }

        // Reorg at LCA = 2: prune >2, add 3' and 4' (same block numbers as
        // the old chain). This works because replace_updates now cleans
        // history bitmaps before re-inserting.
        let b3p = make_block_ref(3, B256::repeat_byte(0xA3), B256::repeat_byte(0x02));
        let b4p = make_block_ref(4, B256::repeat_byte(0xA4), B256::repeat_byte(0xA3));

        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider
                .replace_updates(
                    BlockNumHash::new(2, B256::repeat_byte(0x02)),
                    vec![(b3p, make_nonce_diff(addr, 300)), (b4p, make_nonce_diff(addr, 400))],
                )
                .expect("replace");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Verify: current state has nonce 400
        {
            let tx = db.tx().expect("ro");
            let mut cur = tx.cursor_read::<HashedAccounts>().expect("cursor");
            let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
            assert_eq!(acc.nonce, 400);
        }

        // Verify: changesets exist for blocks 1, 2, 3', 4'
        {
            let tx = db.tx().expect("ro");
            let mut cs = tx.cursor_read::<HashedAccountChangeSets>().expect("cursor");
            assert!(cs.seek_exact(1u64).expect("seek").is_some(), "block 1 changeset");
            assert!(cs.seek_exact(2u64).expect("seek").is_some(), "block 2 changeset");
            assert!(cs.seek_exact(3u64).expect("seek").is_some(), "block 3' changeset");
            assert!(cs.seek_exact(4u64).expect("seek").is_some(), "block 4' changeset");
        }
    }

    // ========================== Unwind tests ==========================

    #[test]
    fn test_unwind_history_to_earliest() {
        let db = setup_db();
        let addr = B256::from([0xBB; 32]);

        // Initialize and set earliest at block 1
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider
                .store_hashed_accounts(vec![(addr, Some(Account::default()))])
                .expect("init");
            provider
                .set_initial_state_anchor(BlockNumHash::new(1, B256::repeat_byte(0x01)))
                .expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Store blocks 2, 3
        let b2 = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));
        let b3 = make_block_ref(3, B256::repeat_byte(0x03), B256::repeat_byte(0x02));
        store_block(&db, b2, make_nonce_diff(addr, 2));
        store_block(&db, b3, make_nonce_diff(addr, 3));

        // Try to unwind to block 1 (= earliest) — should error
        let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
        let unwind_to = BlockWithParent::new(
            B256::repeat_byte(0x01),
            NumHash::new(1, B256::repeat_byte(0x01)),
        );
        let res = provider.unwind_history(unwind_to);
        assert!(
            matches!(res, Err(OpProofsStorageError::UnwindBeyondEarliest { .. })),
            "should error when unwinding to earliest"
        );
    }

    #[test]
    fn test_unwind_history_with_storage() {
        let db = setup_db();

        let addr = B256::from([0xCC; 32]);
        let slot = B256::from([0xDD; 32]);

        // Initialize with account and storage
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider
                .store_hashed_accounts(vec![(addr, Some(Account::default()))])
                .expect("init");
            provider
                .store_hashed_storages(addr, vec![(slot, U256::from(100u64))])
                .expect("init");
            provider
                .set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO))
                .expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Block 1: update both account and storage
        {
            let mut post_state = HashedPostState::default();
            post_state
                .accounts
                .insert(addr, Some(Account { nonce: 1, ..Default::default() }));
            let mut storage = HashedStorage::default();
            storage.storage.insert(slot, U256::from(200u64));
            post_state.storages.insert(addr, storage);
            let diff = BlockStateDiff {
                sorted_trie_updates: TrieUpdates::default().into_sorted(),
                sorted_post_state: post_state.into_sorted(),
            };
            let b1 = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
            store_block(&db, b1, diff);
        }

        // Block 2: update storage again
        {
            let mut post_state = HashedPostState::default();
            post_state
                .accounts
                .insert(addr, Some(Account { nonce: 2, ..Default::default() }));
            let mut storage = HashedStorage::default();
            storage.storage.insert(slot, U256::from(300u64));
            post_state.storages.insert(addr, storage);
            let diff = BlockStateDiff {
                sorted_trie_updates: TrieUpdates::default().into_sorted(),
                sorted_post_state: post_state.into_sorted(),
            };
            let b2 = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));
            store_block(&db, b2, diff);
        }

        // Unwind to block 2 (removes blocks 2+)
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            let unwind_to = BlockWithParent::new(
                B256::repeat_byte(0x01),
                NumHash::new(2, B256::repeat_byte(0x02)),
            );
            provider.unwind_history(unwind_to).expect("unwind");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Verify: account restored to nonce 1, storage restored to 200
        let tx = db.tx().expect("ro");
        let mut acc_cur = tx.cursor_read::<HashedAccounts>().expect("cursor");
        let (_, acc) = acc_cur.seek_exact(addr).expect("seek").expect("exists");
        assert_eq!(acc.nonce, 1, "account should be restored to block 1 state");

        let mut stor_cur = tx.cursor_dup_read::<HashedStorages>().expect("cursor");
        let entry = stor_cur.seek_by_key_subkey(addr, slot).expect("seek").expect("exists");
        assert_eq!(entry.value, U256::from(200u64), "storage should be restored to block 1");
    }

    #[test]
    fn test_unwind_history_with_trie_nodes() {
        let db = setup_db();

        let path1 = Nibbles::from_nibbles_unchecked([0x01]);
        let path2 = Nibbles::from_nibbles_unchecked([0x02]);
        let node1 = BranchNodeCompact::new(0b1, 0, 0, vec![], Some(B256::repeat_byte(0x11)));
        let node2 = BranchNodeCompact::new(0b10, 0, 0, vec![], Some(B256::repeat_byte(0x22)));

        // Initialize with node1 at path1
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider
                .store_account_branches(vec![(path1, Some(node1.clone()))])
                .expect("init");
            provider
                .set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO))
                .expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Block 1: add node2 at path2
        {
            let mut trie_updates = TrieUpdates::default();
            trie_updates.account_nodes.insert(path2, node2.clone());
            let diff = BlockStateDiff {
                sorted_trie_updates: trie_updates.into_sorted(),
                ..Default::default()
            };
            let b1 = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
            store_block(&db, b1, diff);
        }

        // Block 2: overwrite path1
        {
            let mut trie_updates = TrieUpdates::default();
            trie_updates.account_nodes.insert(path1, node2.clone());
            let diff = BlockStateDiff {
                sorted_trie_updates: trie_updates.into_sorted(),
                ..Default::default()
            };
            let b2 = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));
            store_block(&db, b2, diff);
        }

        // Unwind to block 2 (removes blocks 2+)
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            let unwind_to = BlockWithParent::new(
                B256::repeat_byte(0x01),
                NumHash::new(2, B256::repeat_byte(0x02)),
            );
            provider.unwind_history(unwind_to).expect("unwind");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Verify: path1 should be restored to node1, path2 should still have node2 (from block 1)
        let tx = db.tx().expect("ro");
        let mut cur = tx.cursor_read::<AccountsTrie>().expect("cursor");
        let (_, n) = cur.seek_exact(StoredNibbles(path1)).expect("seek").expect("exists");
        assert_eq!(n, node1, "path1 should be restored to original node");
        let (_, n2) = cur.seek_exact(StoredNibbles(path2)).expect("seek").expect("exists");
        assert_eq!(n2, node2, "path2 should still have block 1 node");
    }

    #[test]
    fn test_unwind_history_comprehensive() {
        let db = setup_db();

        let addr1 = B256::from([0x11; 32]);
        let addr2 = B256::from([0x22; 32]);
        let slot1 = B256::from([0xA1; 32]);
        let path1 = Nibbles::from_nibbles_unchecked([0x01]);
        let path2 = Nibbles::from_nibbles_unchecked([0x02]);
        let storage_path1 = Nibbles::from_nibbles_unchecked([0x03]);

        let acc1 = Account { nonce: 1, ..Default::default() };
        let node1 = sample_node();
        let snode1 = sample_node();

        // Initialize
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider
                .store_hashed_accounts(vec![(addr1, Some(acc1))])
                .expect("init");
            provider
                .store_hashed_storages(addr1, vec![(slot1, U256::from(1111u64))])
                .expect("init");
            provider
                .store_account_branches(vec![(path1, Some(node1.clone()))])
                .expect("init");
            provider
                .store_storage_branches(addr1, vec![(storage_path1, Some(snode1.clone()))])
                .expect("init");
            provider
                .set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO))
                .expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Block 1: update all domains
        {
            let mut trie_updates = TrieUpdates::default();
            trie_updates.account_nodes.insert(path1, BranchNodeCompact::default());
            let mut stu = StorageTrieUpdates::default();
            stu.storage_nodes.insert(storage_path1, BranchNodeCompact::default());
            trie_updates.storage_tries.insert(addr1, stu);

            let mut post_state = HashedPostState::default();
            post_state
                .accounts
                .insert(addr1, Some(Account { nonce: 10, ..Default::default() }));
            let mut storage = HashedStorage::default();
            storage.storage.insert(slot1, U256::from(2222u64));
            post_state.storages.insert(addr1, storage);

            let diff = BlockStateDiff {
                sorted_trie_updates: trie_updates.into_sorted(),
                sorted_post_state: post_state.into_sorted(),
            };
            let b1 = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
            store_block(&db, b1, diff);
        }

        // Block 2: more updates
        {
            let mut trie_updates = TrieUpdates::default();
            trie_updates.account_nodes.insert(path2, BranchNodeCompact::default());

            let mut post_state = HashedPostState::default();
            post_state
                .accounts
                .insert(addr2, Some(Account { nonce: 20, ..Default::default() }));

            let diff = BlockStateDiff {
                sorted_trie_updates: trie_updates.into_sorted(),
                sorted_post_state: post_state.into_sorted(),
            };
            let b2 = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));
            store_block(&db, b2, diff);
        }

        // Unwind to block 2 (removes blocks 2+)
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            let unwind_to = BlockWithParent::new(
                B256::repeat_byte(0x01),
                NumHash::new(2, B256::repeat_byte(0x02)),
            );
            provider.unwind_history(unwind_to).expect("unwind");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        let tx = db.tx().expect("ro");

        // Verify account: addr1 should have block 1 state (nonce 10)
        let mut acc_cur = tx.cursor_read::<HashedAccounts>().expect("cursor");
        let (_, acc) = acc_cur.seek_exact(addr1).expect("seek").expect("exists");
        assert_eq!(acc.nonce, 10, "addr1 should have block 1 state");
        // addr2 should not exist (was added in block 2, unwound)
        assert!(acc_cur.seek_exact(addr2).expect("seek").is_none(), "addr2 should be removed");

        // Verify trie: path1 should have block 1 value
        let mut trie_cur = tx.cursor_read::<AccountsTrie>().expect("cursor");
        assert!(trie_cur.seek_exact(StoredNibbles(path1)).expect("seek").is_some());
        // path2 should not exist (added in block 2, unwound)
        assert!(
            trie_cur.seek_exact(StoredNibbles(path2)).expect("seek").is_none(),
            "path2 should be removed"
        );

        // Verify storage: should have block 1 value
        let mut stor_cur = tx.cursor_dup_read::<HashedStorages>().expect("cursor");
        let entry = stor_cur.seek_by_key_subkey(addr1, slot1).expect("seek").expect("exists");
        assert_eq!(entry.value, U256::from(2222u64), "storage should have block 1 value");

        // Verify changesets for blocks 2+ are gone
        let mut cs = tx.cursor_read::<HashedAccountChangeSets>().expect("cursor");
        assert!(cs.seek_exact(1u64).expect("seek").is_some(), "block 1 changeset should remain");
        assert!(
            cs.seek_exact(2u64).expect("seek").is_none(),
            "block 2 changeset should be removed"
        );

        // Verify ProofWindow latest
        let mut pw_cur = tx.cursor_read::<ProofWindow>().expect("cursor");
        let (_, val) =
            pw_cur.seek_exact(ProofWindowKey::LatestBlock).expect("seek").expect("exists");
        assert_eq!(val.number(), 1);
    }

    #[test]
    fn test_unwind_history_empty_chain() {
        let db = setup_db();

        // No blocks stored
        let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
        let unwind_to = BlockWithParent::new(B256::ZERO, NumHash::new(0, B256::ZERO));
        let result = provider.unwind_history(unwind_to);
        assert!(result.is_ok(), "unwinding empty chain should succeed");
    }

    #[test]
    fn test_unwind_history_idempotent() {
        let db = setup_db();
        let addr = B256::from([0xDD; 32]);
        init_state(&db, vec![(addr, Some(Account::default()))]);

        // Store blocks 1, 2, 3
        let b1 = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
        let b2 = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));
        let b3 = make_block_ref(3, B256::repeat_byte(0x03), B256::repeat_byte(0x02));
        store_block(&db, b1, make_nonce_diff(addr, 10));
        store_block(&db, b2, make_nonce_diff(addr, 20));
        store_block(&db, b3, make_nonce_diff(addr, 30));

        // Unwind to block 2
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider.unwind_history(b2).expect("first unwind");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Unwind again (should be idempotent)
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider.unwind_history(b2).expect("second unwind");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Verify state
        let tx = db.tx().expect("ro");
        let mut cur = tx.cursor_read::<HashedAccounts>().expect("cursor");
        let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
        assert_eq!(acc.nonce, 10, "should have block 1 state after unwind to block 2");
    }

    #[test]
    fn test_unwind_history_beyond_latest() {
        let db = setup_db();
        let addr = B256::from([0xEE; 32]);
        init_state(&db, vec![(addr, Some(Account::default()))]);

        // Store blocks 1, 2, 3
        let b1 = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
        let b2 = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));
        let b3 = make_block_ref(3, B256::repeat_byte(0x03), B256::repeat_byte(0x02));
        store_block(&db, b1, make_nonce_diff(addr, 10));
        store_block(&db, b2, make_nonce_diff(addr, 20));
        store_block(&db, b3, make_nonce_diff(addr, 30));

        // Unwind to block 5 (beyond latest) — should be no-op
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            let b5 = make_block_ref(5, B256::repeat_byte(0x05), B256::repeat_byte(0x04));
            provider.unwind_history(b5).expect("unwind");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // All blocks should remain
        let tx = db.tx().expect("ro");
        let mut cur = tx.cursor_read::<HashedAccounts>().expect("cursor");
        let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
        assert_eq!(acc.nonce, 30, "state should be unchanged");

        let mut pw_cur = tx.cursor_read::<ProofWindow>().expect("cursor");
        let (_, val) =
            pw_cur.seek_exact(ProofWindowKey::LatestBlock).expect("seek").expect("exists");
        assert_eq!(val.number(), 3, "latest should be unchanged");
    }
}
