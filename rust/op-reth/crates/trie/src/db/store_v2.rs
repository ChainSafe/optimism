//! V2 MDBX implementation of [`OpProofsStore`](crate::OpProofsStore).
//!
//! This module implements the v2 table schema using **3-table-per-data-type** pattern:
//!
//! | Domain | Current State | ChangeSet | History Bitmap |
//! |--------|--------------|-----------|----------------|
//! | Hashed Accounts | [`V2HashedAccounts`] | [`V2HashedAccountChangeSets`] | [`V2HashedAccountsHistory`] |
//! | Hashed Storages | [`V2HashedStorages`] | [`V2HashedStorageChangeSets`] | [`V2HashedStoragesHistory`] |
//! | Account Trie | [`V2AccountsTrie`] | [`V2AccountTrieChangeSets`] | [`V2AccountsTrieHistory`] |
//! | Storage Trie | [`V2StoragesTrie`] | [`V2StorageTrieChangeSets`] | [`V2StoragesTrieHistory`] |

use super::{BlockNumberHash, ProofWindowKey, Tables, V2ProofWindow};
use crate::{
    BlockStateDiff, OpProofsStorageError, OpProofsStorageResult,
    api::{
        InitialStateAnchor, InitialStateStatus, OpProofsInitProvider, OpProofsProviderRO,
        OpProofsProviderRw, OpProofsStore, WriteCounts,
    },
    db::{
        HashedStorageKey, StorageTrieKey,
        common::ProofWindowValue,
        cursor_v2::{V2AccountCursor, V2AccountTrieCursor, V2StorageCursor, V2StorageTrieCursor},
        models::{
            AccountTrieShardedKey, BlockNumberHashedAddress, HashedAccountBeforeTx,
            HashedAccountShardedKey, HashedStorageShardedKey, StorageTrieShardedKey,
            TrieChangeSetsEntry, V2AccountTrieChangeSets, V2AccountsTrie, V2AccountsTrieHistory,
            V2HashedAccountChangeSets, V2HashedAccounts, V2HashedAccountsHistory,
            V2HashedStorageChangeSets, V2HashedStorages, V2HashedStoragesHistory,
            V2StorageTrieChangeSets, V2StoragesTrie, V2StoragesTrieHistory,
        },
    },
};
use alloy_eips::{BlockNumHash, NumHash, eip1898::BlockWithParent};
use alloy_primitives::{B256, BlockNumber, U256};
#[cfg(feature = "metrics")]
use metrics::{Label, gauge};
use reth_db::{
    BlockNumberList, Database, DatabaseEnv, DatabaseError,
    cursor::{DbCursorRO, DbCursorRW, DbDupCursorRO, DbDupCursorRW},
    mdbx::{DatabaseArguments, init_db_for},
    models::sharded_key::ShardedKey,
    table::Table,
    transaction::{DbTx, DbTxMut},
};
use reth_primitives_traits::{Account, StorageEntry};
use reth_trie::{
    BranchNodeCompact, HashedPostState, HashedPostStateSorted, Nibbles, StorageTrieEntry,
    StoredNibbles, StoredNibblesSubKey,
    updates::{TrieUpdates, TrieUpdatesSorted},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
    path::Path,
    sync::Arc,
};

/// Maximum number of block indices per shard in history bitmap tables.
const NUM_OF_INDICES_IN_SHARD: usize = 2_000;

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
                    let table_db =
                        tx.inner().open_db(Some(table)).wrap_err("Could not open db.")?;

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

        if let Ok(stat) = self.env.stat().map_err(|error| error!(%error, "Failed to read db.stat"))
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
    pub const fn new(tx: TX) -> Self {
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
        let mut cursor = self.tx.cursor_read::<V2ProofWindow>()?;
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

    fn get_proof_window_inner(&self) -> OpProofsStorageResult<ProofWindowValue> {
        let mut cursor = self.tx.cursor_read::<V2ProofWindow>()?;

        let earliest = match cursor.seek_exact(ProofWindowKey::EarliestBlock)? {
            Some((_, val)) => NumHash::new(val.number(), *val.hash()),
            None => return Err(OpProofsStorageError::NoBlocksFound),
        };

        let latest = match cursor.seek_exact(ProofWindowKey::LatestBlock)? {
            Some((_, val)) => NumHash::new(val.number(), *val.hash()),
            None => earliest,
        };

        Ok(ProofWindowValue { earliest, latest })
    }

    fn get_initial_state_anchor_inner(&self) -> OpProofsStorageResult<Option<BlockNumHash>> {
        let mut cur = self.tx.cursor_read::<V2ProofWindow>()?;
        Ok(cur.seek_exact(ProofWindowKey::InitialStateAnchor)?.map(|(_k, v)| v.into()))
    }

    /// Fetch the state diff for a block from changeset tables.
    fn fetch_trie_updates_inner(&self, block_number: u64) -> OpProofsStorageResult<BlockStateDiff> {
        Ok(BlockStateDiff {
            sorted_trie_updates: self.fetch_block_trie_updates(block_number)?.into_sorted(),
            sorted_post_state: self.fetch_block_post_state(block_number)?.into_sorted(),
        })
    }

    /// Reconstruct [`TrieUpdates`] for a block by reading changeset + current state tables.
    fn fetch_block_trie_updates(&self, block_number: u64) -> OpProofsStorageResult<TrieUpdates> {
        let mut updates = TrieUpdates::default();

        // Account trie: read which paths changed, look up their current node.
        let mut acct_state = self.tx.cursor_read::<V2AccountsTrie>()?;
        let mut cs = self.tx.cursor_read::<V2AccountTrieChangeSets>()?;
        let mut walker = cs.walk(Some(block_number))?;
        while let Some(Ok((bn, entry))) = walker.next() {
            if bn != block_number {
                break;
            }
            let path = entry.nibbles.0;
            match acct_state.seek_exact(StoredNibbles(path))?.map(|(_, n)| n) {
                Some(node) => {
                    updates.account_nodes.insert(path, node);
                }
                None => {
                    updates.removed_nodes.insert(path);
                }
            }
        }

        // Storage trie: same pattern, keyed by (block_number, hashed_address).
        let mut stor_state = self.tx.cursor_dup_read::<V2StoragesTrie>()?;
        let mut cs = self.tx.cursor_read::<V2StorageTrieChangeSets>()?;
        let blk_range = BlockNumberHashedAddress((block_number, B256::ZERO))..=
            BlockNumberHashedAddress((block_number, B256::repeat_byte(0xff)));
        let mut walker = cs.walk_range(blk_range)?;
        while let Some(Ok((key, entry))) = walker.next() {
            let hashed_address = key.0.1;
            let subkey = StoredNibblesSubKey(entry.nibbles.0);
            let current_node = stor_state
                .seek_by_key_subkey(hashed_address, subkey.clone())?
                .filter(|e| e.nibbles == subkey)
                .map(|e| e.node);
            let stu = updates.storage_tries.entry(hashed_address).or_default();
            match current_node {
                Some(node) => {
                    stu.storage_nodes.insert(entry.nibbles.0, node);
                }
                None => {
                    stu.removed_nodes.insert(entry.nibbles.0);
                }
            }
        }

        Ok(updates)
    }

    /// Reconstruct [`HashedPostState`] for a block by reading changeset + current state tables.
    fn fetch_block_post_state(&self, block_number: u64) -> OpProofsStorageResult<HashedPostState> {
        let mut post_state = HashedPostState::default();

        // Hashed accounts: read who changed, look up their current account.
        let mut acct_state = self.tx.cursor_read::<V2HashedAccounts>()?;
        let mut cs = self.tx.cursor_read::<V2HashedAccountChangeSets>()?;
        let mut walker = cs.walk(Some(block_number))?;
        while let Some(Ok((bn, entry))) = walker.next() {
            if bn != block_number {
                break;
            }
            let current = acct_state.seek_exact(entry.hashed_address)?.map(|(_, a)| a);
            post_state.accounts.insert(entry.hashed_address, current);
        }

        // Hashed storages: read which slots changed, look up their current value.
        let mut stor_state = self.tx.cursor_dup_read::<V2HashedStorages>()?;
        let mut cs = self.tx.cursor_read::<V2HashedStorageChangeSets>()?;
        let blk_range = BlockNumberHashedAddress((block_number, B256::ZERO))..=
            BlockNumberHashedAddress((block_number, B256::repeat_byte(0xff)));
        let mut walker = cs.walk_range(blk_range)?;
        while let Some(Ok((key, entry))) = walker.next() {
            let hashed_address = key.0.1;
            let current_value = stor_state
                .seek_by_key_subkey(hashed_address, entry.key)?
                .filter(|e| e.key == entry.key)
                .map(|e| e.value)
                .unwrap_or(U256::ZERO);
            post_state
                .storages
                .entry(hashed_address)
                .or_default()
                .storage
                .insert(entry.key, current_value);
        }

        Ok(post_state)
    }
}

// =============================================================================
// Read-write helpers
// =============================================================================

/// Collector for batched history bitmap appends.
///
/// Instead of performing one `seek_exact` + decode + push + re-encode +
/// `upsert` per entry (the old inline approach), we collect `(key, block)`
/// pairs and flush them at the end of a batch.  For keys that appear in
/// multiple blocks within the batch this turns N round-trips into 1.
/// The `BTreeMap` also gives sorted iteration, so cursor seeks during
/// flush are sequential (cache-friendly).
#[derive(Default)]
struct HistoryCollector {
    account_trie: BTreeMap<StoredNibbles, Vec<BlockNumber>>,
    storage_trie: BTreeMap<(B256, StoredNibbles), Vec<BlockNumber>>,
    hashed_accounts: BTreeMap<B256, Vec<BlockNumber>>,
    hashed_storages: BTreeMap<(B256, B256), Vec<BlockNumber>>,
}

/// Pre-opened write cursors for the 8 tables touched by
/// [`MdbxProofsProviderV2::store_block_updates`].
///
/// Avoids re-opening cursors on every block in a batch — each `mdbx_cursor_open`
/// has measurable overhead, and for a 5-block batch this turns 40 cursor
/// open+drop cycles into 8.
struct WriteCursors<TX: DbTxMut + DbTx> {
    account_trie_state: <TX as DbTxMut>::CursorMut<V2AccountsTrie>,
    account_trie_cs: <TX as DbTxMut>::DupCursorMut<V2AccountTrieChangeSets>,
    storage_trie_state: <TX as DbTxMut>::DupCursorMut<V2StoragesTrie>,
    storage_trie_cs: <TX as DbTxMut>::DupCursorMut<V2StorageTrieChangeSets>,
    hashed_accounts_state: <TX as DbTxMut>::CursorMut<V2HashedAccounts>,
    hashed_accounts_cs: <TX as DbTxMut>::DupCursorMut<V2HashedAccountChangeSets>,
    hashed_storages_state: <TX as DbTxMut>::DupCursorMut<V2HashedStorages>,
    hashed_storages_cs: <TX as DbTxMut>::DupCursorMut<V2HashedStorageChangeSets>,
}

impl<TX: DbTxMut + DbTx> WriteCursors<TX> {
    fn new(tx: &TX) -> OpProofsStorageResult<Self> {
        Ok(Self {
            account_trie_state: tx.cursor_write::<V2AccountsTrie>()?,
            account_trie_cs: tx.cursor_dup_write::<V2AccountTrieChangeSets>()?,
            storage_trie_state: tx.cursor_dup_write::<V2StoragesTrie>()?,
            storage_trie_cs: tx.cursor_dup_write::<V2StorageTrieChangeSets>()?,
            hashed_accounts_state: tx.cursor_write::<V2HashedAccounts>()?,
            hashed_accounts_cs: tx.cursor_dup_write::<V2HashedAccountChangeSets>()?,
            hashed_storages_state: tx.cursor_dup_write::<V2HashedStorages>()?,
            hashed_storages_cs: tx.cursor_dup_write::<V2HashedStorageChangeSets>()?,
        })
    }
}

/// Append multiple block numbers to a sharded history bitmap in a single
/// seek+decode+push-all+upsert round-trip.
///
/// This is the batched equivalent of
/// [`MdbxProofsProviderV2::append_history_index_with_cursor`].
fn append_history_indices_batched<T>(
    cursor: &mut (impl DbCursorRO<T> + DbCursorRW<T>),
    block_numbers: &[BlockNumber],
    sharded_key_factory: impl Fn(BlockNumber) -> T::Key,
) -> OpProofsStorageResult<()>
where
    T: Table<Value = BlockNumberList>,
    T::Key: Clone,
{
    if block_numbers.is_empty() {
        return Ok(());
    }

    let last_key = sharded_key_factory(u64::MAX);

    let mut last_shard = cursor
        .seek_exact(last_key.clone())?
        .map(|(_, list)| list)
        .unwrap_or_else(BlockNumberList::empty);

    for &bn in block_numbers {
        last_shard
            .push(bn)
            .map_err(|e| DatabaseError::Other(format!("IntegerList push error: {e}")))?;
    }

    // Fast path: fits in one shard
    if last_shard.len() <= NUM_OF_INDICES_IN_SHARD as u64 {
        cursor.upsert(last_key, &last_shard)?;
        return Ok(());
    }

    // Slow path: rechunk
    if cursor.seek_exact(last_key)?.is_some() {
        cursor.delete_current()?;
    }

    let all_values: Vec<u64> = last_shard.iter().collect();
    let total_chunks = all_values.chunks(NUM_OF_INDICES_IN_SHARD).len();

    for (i, chunk) in all_values.chunks(NUM_OF_INDICES_IN_SHARD).enumerate() {
        let shard = BlockNumberList::new_pre_sorted(chunk.iter().copied());
        let key = if i < total_chunks - 1 {
            sharded_key_factory(*chunk.last().expect("non-empty chunk"))
        } else {
            sharded_key_factory(u64::MAX)
        };
        cursor.upsert(key, &shard)?;
    }

    Ok(())
}

impl<TX: DbTxMut + DbTx> MdbxProofsProviderV2<TX> {
    fn set_earliest_block_number_inner(
        &self,
        block_number: u64,
        hash: B256,
    ) -> OpProofsStorageResult<()> {
        let mut cursor = self.tx.cursor_write::<V2ProofWindow>()?;
        cursor.upsert(ProofWindowKey::EarliestBlock, &BlockNumberHash::new(block_number, hash))?;
        Ok(())
    }

    fn set_latest_block_number_inner(
        &self,
        block_number: u64,
        hash: B256,
    ) -> OpProofsStorageResult<()> {
        let mut cursor = self.tx.cursor_write::<V2ProofWindow>()?;
        cursor.upsert(ProofWindowKey::LatestBlock, &BlockNumberHash::new(block_number, hash))?;
        Ok(())
    }

    /// Prune-specific history removal: for a given logical key, seek its first
    /// history shard and walk forward, removing all block numbers that fall
    /// within `range`.  Requires only **one seek per unique key** (instead
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
            let filtered: Vec<u64> = list.iter().filter(|&bn| !range.contains(&bn)).collect();

            if filtered.is_empty() {
                // Entire shard pruned — delete and advance.
                cursor.delete_current()?;
                entry = cursor.current()?;
            } else if filtered.len() < original_len {
                // Partial prune — update shard and advance.
                let new_list = BlockNumberList::new_pre_sorted(filtered);
                cursor.upsert(key, &new_list)?;
                entry = cursor.next()?;
            } else {
                // No blocks in this shard were in range.
                // If the shard's lowest block is past our range, stop.
                if list.iter().next().is_none_or(|first| first > *range.end()) {
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
    /// each changeset table is walked **once**, entries are
    /// deduplicated by key into a `BTreeMap<key, BTreeSet<block_number>>`,
    /// and then each unique key's bitmap shard(s) are edited in a single
    /// batch operation through a **reused cursor**.
    /// Single forward scan over account-trie changesets for `range`.
    ///
    /// In one pass: restores the pre-range state for every affected path,
    /// collects the set of affected paths for history bitmap removal, and
    /// deletes the changeset entries.
    ///
    /// Correctness: the first occurrence of each path in a forward scan is the
    /// *smallest* block number in the range, whose old-value is exactly the
    /// state before the entire range — the value we need to restore.
    fn unwind_and_collect_account_trie(
        &self,
        range: &std::ops::RangeInclusive<u64>,
    ) -> OpProofsStorageResult<BTreeSet<StoredNibbles>> {
        let mut restorations: BTreeMap<StoredNibbles, Option<BranchNodeCompact>> = BTreeMap::new();
        {
            let mut cs = self.tx.cursor_dup_write::<V2AccountTrieChangeSets>()?;
            let mut entry = cs.seek(*range.start())?;
            while let Some((block_num, val)) = entry {
                if !range.contains(&block_num) {
                    break;
                }
                let path = StoredNibbles(val.nibbles.0);
                restorations.entry(path).or_insert(val.node);
                while let Some((_, val)) = cs.next_dup()? {
                    restorations.entry(StoredNibbles(val.nibbles.0)).or_insert(val.node);
                }
                cs.delete_current_duplicates()?;
                entry = cs.current()?;
            }
        }
        let mut state = self.tx.cursor_write::<V2AccountsTrie>()?;
        for (path, old_node) in &restorations {
            match old_node {
                Some(node) => state.upsert(path.clone(), node)?,
                None => {
                    if state.seek_exact(path.clone())?.is_some() {
                        state.delete_current()?;
                    }
                }
            }
        }
        Ok(restorations.into_keys().collect())
    }

    /// Single forward scan over storage-trie changesets for `range`.
    ///
    /// See [`Self::unwind_and_collect_account_trie`] for the correctness argument.
    fn unwind_and_collect_storage_trie(
        &self,
        range: &std::ops::RangeInclusive<u64>,
    ) -> OpProofsStorageResult<BTreeSet<(B256, StoredNibbles)>> {
        let mut restorations: BTreeMap<(B256, StoredNibblesSubKey), Option<BranchNodeCompact>> =
            BTreeMap::new();
        {
            let mut cs = self.tx.cursor_dup_write::<V2StorageTrieChangeSets>()?;
            let start = BlockNumberHashedAddress((*range.start(), B256::ZERO));
            let end = BlockNumberHashedAddress((*range.end(), B256::repeat_byte(0xff)));
            let mut entry = cs.seek(start)?;
            while let Some((key, val)) = entry {
                if key > end || key < start {
                    break;
                }
                restorations.entry((key.0.1, val.nibbles.clone())).or_insert(val.node);
                while let Some((k, val)) = cs.next_dup()? {
                    restorations.entry((k.0.1, val.nibbles.clone())).or_insert(val.node);
                }
                cs.delete_current_duplicates()?;
                entry = cs.current()?;
            }
        }
        let mut state = self.tx.cursor_dup_write::<V2StoragesTrie>()?;
        for ((addr, subkey), old_node) in &restorations {
            if state
                .seek_by_key_subkey(*addr, subkey.clone())?
                .filter(|e| e.nibbles == *subkey)
                .is_some()
            {
                state.delete_current()?;
            }
            if let Some(node) = old_node {
                state.upsert(
                    *addr,
                    &StorageTrieEntry { nibbles: subkey.clone(), node: node.clone() },
                )?;
            }
        }
        Ok(restorations.into_keys().map(|(addr, subkey)| (addr, StoredNibbles(subkey.0))).collect())
    }

    /// Single forward scan over hashed-account changesets for `range`.
    ///
    /// See [`Self::unwind_and_collect_account_trie`] for the correctness argument.
    fn unwind_and_collect_hashed_accounts(
        &self,
        range: &std::ops::RangeInclusive<u64>,
    ) -> OpProofsStorageResult<BTreeSet<B256>> {
        let mut restorations: BTreeMap<B256, Option<Account>> = BTreeMap::new();
        {
            let mut cs = self.tx.cursor_dup_write::<V2HashedAccountChangeSets>()?;
            let mut entry = cs.seek(*range.start())?;
            while let Some((block_num, val)) = entry {
                if !range.contains(&block_num) {
                    break;
                }
                restorations.entry(val.hashed_address).or_insert(val.info);
                while let Some((_, val)) = cs.next_dup()? {
                    restorations.entry(val.hashed_address).or_insert(val.info);
                }
                cs.delete_current_duplicates()?;
                entry = cs.current()?;
            }
        }
        let mut state = self.tx.cursor_write::<V2HashedAccounts>()?;
        for (addr, old_account) in &restorations {
            match old_account {
                Some(account) => state.upsert(*addr, account)?,
                None => {
                    if state.seek_exact(*addr)?.is_some() {
                        state.delete_current()?;
                    }
                }
            }
        }
        Ok(restorations.into_keys().collect())
    }

    /// Single forward scan over hashed-storage changesets for `range`.
    ///
    /// See [`Self::unwind_and_collect_account_trie`] for the correctness argument.
    fn unwind_and_collect_hashed_storages(
        &self,
        range: &std::ops::RangeInclusive<u64>,
    ) -> OpProofsStorageResult<BTreeSet<(B256, B256)>> {
        let mut restorations: BTreeMap<(B256, B256), U256> = BTreeMap::new();
        {
            let mut cs = self.tx.cursor_dup_write::<V2HashedStorageChangeSets>()?;
            let start = BlockNumberHashedAddress((*range.start(), B256::ZERO));
            let end = BlockNumberHashedAddress((*range.end(), B256::repeat_byte(0xff)));
            let mut entry = cs.seek(start)?;
            while let Some((key, val)) = entry {
                if key > end || key < start {
                    break;
                }
                restorations.entry((key.0.1, val.key)).or_insert(val.value);
                while let Some((k, val)) = cs.next_dup()? {
                    restorations.entry((k.0.1, val.key)).or_insert(val.value);
                }
                cs.delete_current_duplicates()?;
                entry = cs.current()?;
            }
        }
        let mut state = self.tx.cursor_dup_write::<V2HashedStorages>()?;
        for ((addr, slot), old_value) in &restorations {
            if state.seek_by_key_subkey(*addr, *slot)?.filter(|e| e.key == *slot).is_some() {
                state.delete_current()?;
            }
            if *old_value != U256::ZERO {
                state.upsert(*addr, &StorageEntry { key: *slot, value: *old_value })?;
            }
        }
        Ok(restorations.into_keys().collect())
    }

    /// Core write logic for a single block.
    ///
    /// Delegates each data domain to a focused helper, then assembles counts.
    fn store_block_updates(
        &self,
        block_number: BlockNumber,
        block_state_diff: BlockStateDiff,
        collector: &mut HistoryCollector,
        cursors: &mut WriteCursors<TX>,
    ) -> OpProofsStorageResult<WriteCounts> {
        let BlockStateDiff { sorted_trie_updates, sorted_post_state } = block_state_diff;
        Ok(WriteCounts {
            account_trie_updates_written_total: Self::write_account_trie(
                block_number,
                &sorted_trie_updates,
                cursors,
                collector,
            )?,
            storage_trie_updates_written_total: Self::write_storage_trie(
                block_number,
                &sorted_trie_updates,
                cursors,
                collector,
            )?,
            hashed_accounts_written_total: Self::write_hashed_accounts(
                block_number,
                &sorted_post_state,
                cursors,
                collector,
            )?,
            hashed_storages_written_total: Self::write_hashed_storages(
                block_number,
                &sorted_post_state,
                cursors,
                collector,
            )?,
        })
    }

    /// Write account trie branch-node updates for one block.
    ///
    /// For each changed path: save the old node to the changeset, record the
    /// block number in the history bitmap collector, then apply the new value
    /// (upsert or delete) to the current-state table.
    fn write_account_trie(
        block_number: BlockNumber,
        updates: &TrieUpdatesSorted,
        cursors: &mut WriteCursors<TX>,
        collector: &mut HistoryCollector,
    ) -> OpProofsStorageResult<u64> {
        let state_cursor = &mut cursors.account_trie_state;
        let cs_cursor = &mut cursors.account_trie_cs;
        let mut count = 0u64;

        for (nibbles, maybe_node) in updates.account_nodes_ref() {
            let stored = StoredNibbles(*nibbles);

            let old_entry = state_cursor.seek_exact(stored.clone())?;
            let old_node = old_entry.as_ref().map(|(_, node)| node.clone());
            let had_old = old_entry.is_some();

            cs_cursor.append_dup(
                block_number,
                TrieChangeSetsEntry { nibbles: StoredNibblesSubKey(*nibbles), node: old_node },
            )?;
            collector.account_trie.entry(stored.clone()).or_default().push(block_number);

            match maybe_node {
                Some(node) => state_cursor.upsert(stored, node)?,
                None => {
                    if had_old {
                        state_cursor.delete_current()?;
                    }
                }
            }
            count += 1;
        }
        Ok(count)
    }

    /// Write storage trie branch-node updates for one block.
    ///
    /// Handles the `is_deleted` wipe path (snapshot all existing nodes into
    /// the changeset before clearing) as well as per-node updates.
    fn write_storage_trie(
        block_number: BlockNumber,
        updates: &TrieUpdatesSorted,
        cursors: &mut WriteCursors<TX>,
        collector: &mut HistoryCollector,
    ) -> OpProofsStorageResult<u64> {
        let state_cursor = &mut cursors.storage_trie_state;
        let cs_cursor = &mut cursors.storage_trie_cs;
        let mut count = 0u64;

        for (hashed_address, nodes) in updates.storage_tries_ref() {
            let cs_key = BlockNumberHashedAddress((block_number, *hashed_address));

            if nodes.is_deleted {
                // Snapshot all existing nodes into the changeset before wiping.
                if let Some((_key, first_entry)) = state_cursor.seek_exact(*hashed_address)? {
                    cs_cursor.append_dup(
                        cs_key,
                        TrieChangeSetsEntry {
                            nibbles: first_entry.nibbles.clone(),
                            node: Some(first_entry.node.clone()),
                        },
                    )?;
                    collector
                        .storage_trie
                        .entry((*hashed_address, StoredNibbles(first_entry.nibbles.0)))
                        .or_default()
                        .push(block_number);

                    while let Some((_, entry)) = state_cursor.next_dup()? {
                        cs_cursor.append_dup(
                            cs_key,
                            TrieChangeSetsEntry {
                                nibbles: entry.nibbles.clone(),
                                node: Some(entry.node.clone()),
                            },
                        )?;
                        collector
                            .storage_trie
                            .entry((*hashed_address, StoredNibbles(entry.nibbles.0)))
                            .or_default()
                            .push(block_number);
                    }

                    if state_cursor.seek_exact(*hashed_address)?.is_some() {
                        state_cursor.delete_current_duplicates()?;
                    }
                }
                count += 1;
            }

            for (nibbles, maybe_node) in nodes.storage_nodes_ref() {
                let subkey = StoredNibblesSubKey(*nibbles);

                // Seek positions the cursor — reused for the delete below.
                let old_entry = state_cursor
                    .seek_by_key_subkey(*hashed_address, subkey.clone())?
                    .filter(|e| e.nibbles == subkey);
                let had_old = old_entry.is_some();
                let old_node = old_entry.map(|e| e.node);

                cs_cursor.append_dup(
                    cs_key,
                    TrieChangeSetsEntry { nibbles: subkey.clone(), node: old_node },
                )?;
                collector
                    .storage_trie
                    .entry((*hashed_address, StoredNibbles(*nibbles)))
                    .or_default()
                    .push(block_number);

                match maybe_node {
                    Some(node) => {
                        if had_old {
                            state_cursor.delete_current()?;
                        }
                        state_cursor.upsert(
                            *hashed_address,
                            &StorageTrieEntry { nibbles: subkey, node: node.clone() },
                        )?;
                    }
                    None => {
                        if had_old {
                            state_cursor.delete_current()?;
                        }
                    }
                }
                count += 1;
            }
        }
        Ok(count)
    }

    /// Write hashed-account updates for one block.
    ///
    /// For each changed account: save the old account to the changeset, record
    /// the block number in the history bitmap collector, then apply the new
    /// value (upsert or delete) to the current-state table.
    fn write_hashed_accounts(
        block_number: BlockNumber,
        post_state: &HashedPostStateSorted,
        cursors: &mut WriteCursors<TX>,
        collector: &mut HistoryCollector,
    ) -> OpProofsStorageResult<u64> {
        let state_cursor = &mut cursors.hashed_accounts_state;
        let cs_cursor = &mut cursors.hashed_accounts_cs;
        let mut count = 0u64;

        for (hashed_address, maybe_account) in &post_state.accounts {
            let old_entry = state_cursor.seek_exact(*hashed_address)?;
            let old_account = old_entry.as_ref().map(|(_, acc)| *acc);
            let had_old = old_entry.is_some();

            cs_cursor.append_dup(
                block_number,
                HashedAccountBeforeTx::new(*hashed_address, old_account),
            )?;
            collector.hashed_accounts.entry(*hashed_address).or_default().push(block_number);

            match maybe_account {
                Some(account) => state_cursor.upsert(*hashed_address, account)?,
                None => {
                    if had_old {
                        state_cursor.delete_current()?;
                    }
                }
            }
            count += 1;
        }
        Ok(count)
    }

    /// Write hashed-storage updates for one block.
    ///
    /// Handles the `is_wiped` path (snapshot all existing slots into the
    /// changeset before clearing) as well as per-slot updates.
    fn write_hashed_storages(
        block_number: BlockNumber,
        post_state: &HashedPostStateSorted,
        cursors: &mut WriteCursors<TX>,
        collector: &mut HistoryCollector,
    ) -> OpProofsStorageResult<u64> {
        let state_cursor = &mut cursors.hashed_storages_state;
        let cs_cursor = &mut cursors.hashed_storages_cs;
        let mut count = 0u64;

        for (hashed_address, storage) in &post_state.storages {
            let cs_key = BlockNumberHashedAddress((block_number, *hashed_address));

            if storage.is_wiped() {
                // Snapshot all existing slots into the changeset before wiping.
                // Track which slots were recorded so the per-slot loop below
                // doesn't double-append them.
                let mut wiped_slots = alloy_primitives::map::B256Set::default();

                if let Some(entry) = state_cursor.seek_by_key_subkey(*hashed_address, B256::ZERO)? {
                    cs_cursor.append_dup(cs_key, entry)?;
                    collector
                        .hashed_storages
                        .entry((*hashed_address, entry.key))
                        .or_default()
                        .push(block_number);
                    wiped_slots.insert(entry.key);

                    while let Some(entry) = state_cursor.next_dup_val()? {
                        cs_cursor.append_dup(cs_key, entry)?;
                        collector
                            .hashed_storages
                            .entry((*hashed_address, entry.key))
                            .or_default()
                            .push(block_number);
                        wiped_slots.insert(entry.key);
                    }

                    if state_cursor.seek_exact(*hashed_address)?.is_some() {
                        state_cursor.delete_current_duplicates()?;
                    }
                }

                // Write new slots. Slots not seen during the wipe get a zero
                // old-value entry in the changeset.
                for (storage_key, value) in storage.storage_slots_ref() {
                    if !wiped_slots.contains(storage_key) {
                        cs_cursor.append_dup(
                            cs_key,
                            StorageEntry { key: *storage_key, value: U256::ZERO },
                        )?;
                        collector
                            .hashed_storages
                            .entry((*hashed_address, *storage_key))
                            .or_default()
                            .push(block_number);
                    }
                    if *value != U256::ZERO {
                        state_cursor.upsert(
                            *hashed_address,
                            &StorageEntry { key: *storage_key, value: *value },
                        )?;
                    }
                    count += 1;
                }
            } else {
                for (storage_key, value) in storage.storage_slots_ref() {
                    // Seek positions the cursor — reused for the delete below.
                    let old_entry = state_cursor
                        .seek_by_key_subkey(*hashed_address, *storage_key)?
                        .filter(|e| e.key == *storage_key);
                    let had_old = old_entry.is_some();
                    let old_value = old_entry.map(|e| e.value).unwrap_or(U256::ZERO);

                    cs_cursor
                        .append_dup(cs_key, StorageEntry { key: *storage_key, value: old_value })?;
                    collector
                        .hashed_storages
                        .entry((*hashed_address, *storage_key))
                        .or_default()
                        .push(block_number);

                    if had_old {
                        state_cursor.delete_current()?;
                    }
                    if *value != U256::ZERO {
                        state_cursor.upsert(
                            *hashed_address,
                            &StorageEntry { key: *storage_key, value: *value },
                        )?;
                    }
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    /// Flush all collected history bitmap entries to the database.
    ///
    /// For each unique key, performs a single `seek_exact` + decode +
    /// push-all-block-numbers + re-encode + `upsert` instead of doing
    /// that per-entry.  The `BTreeMap` iteration order ensures cursor
    /// seeks are sequential within each table.
    fn flush_collected_history(&self, collector: HistoryCollector) -> OpProofsStorageResult<()> {
        macro_rules! flush {
            ($table:ty, $entries:expr, $key_fn:expr) => {
                if !$entries.is_empty() {
                    let mut cursor = self.tx.cursor_write::<$table>()?;
                    for (key, blocks) in $entries {
                        append_history_indices_batched::<$table>(
                            &mut cursor,
                            &blocks,
                            |highest| $key_fn(key.clone(), highest),
                        )?;
                    }
                }
            };
        }

        flush!(V2AccountsTrieHistory, collector.account_trie, |path, highest| {
            AccountTrieShardedKey::new(path, highest)
        });
        flush!(V2StoragesTrieHistory, collector.storage_trie, |(addr, path), highest| {
            StorageTrieShardedKey::new(addr, path, highest)
        });
        flush!(V2HashedAccountsHistory, collector.hashed_accounts, |addr, highest| {
            HashedAccountShardedKey::new(addr, highest)
        });
        flush!(V2HashedStoragesHistory, collector.hashed_storages, |(addr, slot), highest| {
            HashedStorageShardedKey {
                hashed_address: addr,
                sharded_key: ShardedKey::new(slot, highest),
            }
        });

        Ok(())
    }

    /// Validate block ordering and return the block number.
    fn validate_block_order(&self, block_ref: &BlockWithParent) -> OpProofsStorageResult<()> {
        let block_number = block_ref.block.number;

        let proof_window = self.get_proof_window_inner()?;

        if proof_window.latest.hash != block_ref.parent {
            return Err(OpProofsStorageError::OutOfOrder {
                block_number,
                parent_block_hash: block_ref.parent,
                latest_block_hash: proof_window.latest.hash,
            });
        }

        Ok(())
    }

    /// Phase A: delete all account-trie changeset entries in `range`, returning the
    /// set of nibble paths that were affected (used in Phase B for history pruning).
    fn prune_account_trie_changesets(
        &self,
        range: &std::ops::RangeInclusive<u64>,
        counts: &mut WriteCounts,
    ) -> OpProofsStorageResult<BTreeSet<StoredNibbles>> {
        let mut keys: BTreeSet<StoredNibbles> = BTreeSet::new();
        let mut cursor = self.tx.cursor_dup_write::<V2AccountTrieChangeSets>()?;
        let mut entry = cursor.seek(*range.start())?;
        while let Some((block_num, first_val)) = entry {
            if block_num > *range.end() {
                break;
            }
            counts.account_trie_updates_written_total += 1;
            keys.insert(StoredNibbles(first_val.nibbles.0));
            while let Some((_, val)) = cursor.next_dup()? {
                counts.account_trie_updates_written_total += 1;
                keys.insert(StoredNibbles(val.nibbles.0));
            }
            cursor.delete_current_duplicates()?;
            entry = cursor.current()?;
        }
        Ok(keys)
    }

    /// Phase A: delete all storage-trie changeset entries in `range`, returning the
    /// set of `(hashed_address, nibbles)` pairs that were affected.
    fn prune_storage_trie_changesets(
        &self,
        range: &std::ops::RangeInclusive<u64>,
        counts: &mut WriteCounts,
    ) -> OpProofsStorageResult<BTreeSet<(B256, StoredNibbles)>> {
        let mut keys: BTreeSet<(B256, StoredNibbles)> = BTreeSet::new();
        let mut cursor = self.tx.cursor_dup_write::<V2StorageTrieChangeSets>()?;
        let start = BlockNumberHashedAddress((*range.start(), B256::ZERO));
        let end = BlockNumberHashedAddress((*range.end(), B256::repeat_byte(0xff)));
        let mut entry = cursor.seek(start)?;
        while let Some((key, first_val)) = entry {
            if key > end {
                break;
            }
            counts.storage_trie_updates_written_total += 1;
            keys.insert((key.0.1, StoredNibbles(first_val.nibbles.0)));
            while let Some((k, val)) = cursor.next_dup()? {
                counts.storage_trie_updates_written_total += 1;
                keys.insert((k.0.1, StoredNibbles(val.nibbles.0)));
            }
            cursor.delete_current_duplicates()?;
            entry = cursor.current()?;
        }
        Ok(keys)
    }

    /// Phase A: delete all hashed-account changeset entries in `range`, returning the
    /// set of hashed addresses that were affected.
    fn prune_hashed_account_changesets(
        &self,
        range: &std::ops::RangeInclusive<u64>,
        counts: &mut WriteCounts,
    ) -> OpProofsStorageResult<BTreeSet<B256>> {
        let mut keys: BTreeSet<B256> = BTreeSet::new();
        let mut cursor = self.tx.cursor_dup_write::<V2HashedAccountChangeSets>()?;
        let mut entry = cursor.seek(*range.start())?;
        while let Some((block_num, first_val)) = entry {
            if block_num > *range.end() {
                break;
            }
            counts.hashed_accounts_written_total += 1;
            keys.insert(first_val.hashed_address);
            while let Some((_, val)) = cursor.next_dup()? {
                counts.hashed_accounts_written_total += 1;
                keys.insert(val.hashed_address);
            }
            cursor.delete_current_duplicates()?;
            entry = cursor.current()?;
        }
        Ok(keys)
    }

    /// Phase A: delete all hashed-storage changeset entries in `range`, returning the
    /// set of `(hashed_address, storage_key)` pairs that were affected.
    fn prune_hashed_storage_changesets(
        &self,
        range: &std::ops::RangeInclusive<u64>,
        counts: &mut WriteCounts,
    ) -> OpProofsStorageResult<BTreeSet<(B256, B256)>> {
        let mut keys: BTreeSet<(B256, B256)> = BTreeSet::new();
        let mut cursor = self.tx.cursor_dup_write::<V2HashedStorageChangeSets>()?;
        let start = BlockNumberHashedAddress((*range.start(), B256::ZERO));
        let end = BlockNumberHashedAddress((*range.end(), B256::repeat_byte(0xff)));
        let mut entry = cursor.seek(start)?;
        while let Some((key, first_val)) = entry {
            if key > end {
                break;
            }
            counts.hashed_storages_written_total += 1;
            keys.insert((key.0.1, first_val.key));
            while let Some((k, val)) = cursor.next_dup()? {
                counts.hashed_storages_written_total += 1;
                keys.insert((k.0.1, val.key));
            }
            cursor.delete_current_duplicates()?;
            entry = cursor.current()?;
        }
        Ok(keys)
    }

    /// Phase B: remove `range` block numbers from account-trie history bitmaps for the
    /// given nibble paths.
    fn prune_account_trie_history(
        &self,
        range: &std::ops::RangeInclusive<u64>,
        keys: &BTreeSet<StoredNibbles>,
    ) -> OpProofsStorageResult<()> {
        let mut cursor = self.tx.cursor_write::<V2AccountsTrieHistory>()?;
        for nibbles in keys {
            Self::prune_history_range_for_key(
                &mut cursor,
                range,
                AccountTrieShardedKey::new(nibbles.clone(), 0),
                |k| k.key == *nibbles,
            )?;
        }
        Ok(())
    }

    /// Phase B: remove `range` block numbers from storage-trie history bitmaps for the
    /// given `(hashed_address, nibbles)` pairs.
    fn prune_storage_trie_history(
        &self,
        range: &std::ops::RangeInclusive<u64>,
        keys: &BTreeSet<(B256, StoredNibbles)>,
    ) -> OpProofsStorageResult<()> {
        let mut cursor = self.tx.cursor_write::<V2StoragesTrieHistory>()?;
        for (hashed_address, nibbles) in keys {
            Self::prune_history_range_for_key(
                &mut cursor,
                range,
                StorageTrieShardedKey::new(*hashed_address, nibbles.clone(), 0),
                |k| k.hashed_address == *hashed_address && k.key == *nibbles,
            )?;
        }
        Ok(())
    }

    /// Phase B: remove `range` block numbers from hashed-account history bitmaps for the
    /// given hashed addresses.
    fn prune_hashed_account_history(
        &self,
        range: &std::ops::RangeInclusive<u64>,
        keys: &BTreeSet<B256>,
    ) -> OpProofsStorageResult<()> {
        let mut cursor = self.tx.cursor_write::<V2HashedAccountsHistory>()?;
        for addr in keys {
            Self::prune_history_range_for_key(
                &mut cursor,
                range,
                HashedAccountShardedKey::new(*addr, 0),
                |k| k.0.key == *addr,
            )?;
        }
        Ok(())
    }

    /// Phase B: remove `range` block numbers from hashed-storage history bitmaps for the
    /// given `(hashed_address, storage_key)` pairs.
    fn prune_hashed_storage_history(
        &self,
        range: &std::ops::RangeInclusive<u64>,
        keys: &BTreeSet<(B256, B256)>,
    ) -> OpProofsStorageResult<()> {
        let mut cursor = self.tx.cursor_write::<V2HashedStoragesHistory>()?;
        for (hashed_address, storage_key) in keys {
            Self::prune_history_range_for_key(
                &mut cursor,
                range,
                HashedStorageShardedKey {
                    hashed_address: *hashed_address,
                    sharded_key: ShardedKey::new(*storage_key, 0),
                },
                |k| k.hashed_address == *hashed_address && k.sharded_key.key == *storage_key,
            )?;
        }
        Ok(())
    }
}

impl<TX: DbTx + Send + Sync + Debug + 'static> OpProofsProviderRO for MdbxProofsProviderV2<TX> {
    type StorageTrieCursor<'tx>
        = V2StorageTrieCursor<
        TX::DupCursor<V2StoragesTrie>,
        TX::Cursor<V2StoragesTrieHistory>,
        TX::DupCursor<V2StorageTrieChangeSets>,
    >
    where
        Self: 'tx,
        TX: 'tx;

    type AccountTrieCursor<'tx>
        = V2AccountTrieCursor<
        TX::Cursor<V2AccountsTrie>,
        TX::Cursor<V2AccountsTrieHistory>,
        TX::DupCursor<V2AccountTrieChangeSets>,
    >
    where
        Self: 'tx,
        TX: 'tx;

    type StorageCursor<'tx>
        = V2StorageCursor<
        TX::DupCursor<V2HashedStorages>,
        TX::Cursor<V2HashedStoragesHistory>,
        TX::DupCursor<V2HashedStorageChangeSets>,
    >
    where
        Self: 'tx,
        TX: 'tx;

    type AccountHashedCursor<'tx>
        = V2AccountCursor<
        TX::Cursor<V2HashedAccounts>,
        TX::Cursor<V2HashedAccountsHistory>,
        TX::DupCursor<V2HashedAccountChangeSets>,
    >
    where
        Self: 'tx,
        TX: 'tx;

    fn get_earliest_block_number(&self) -> OpProofsStorageResult<Option<(u64, B256)>> {
        self.get_block_number_hash_inner(ProofWindowKey::EarliestBlock)
    }

    fn get_latest_block_number(&self) -> OpProofsStorageResult<Option<(u64, B256)>> {
        let mut cursor = self.tx.cursor_read::<V2ProofWindow>()?;
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
            self.tx.cursor_dup_read::<V2StoragesTrie>()?,
            self.tx.cursor_read::<V2StoragesTrieHistory>()?,
            self.tx.cursor_read::<V2StoragesTrieHistory>()?,
            self.tx.cursor_dup_read::<V2StorageTrieChangeSets>()?,
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
            self.tx.cursor_read::<V2AccountsTrie>()?,
            self.tx.cursor_read::<V2AccountsTrieHistory>()?,
            self.tx.cursor_read::<V2AccountsTrieHistory>()?,
            self.tx.cursor_dup_read::<V2AccountTrieChangeSets>()?,
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
            self.tx.cursor_dup_read::<V2HashedStorages>()?,
            self.tx.cursor_read::<V2HashedStoragesHistory>()?,
            self.tx.cursor_read::<V2HashedStoragesHistory>()?,
            self.tx.cursor_dup_read::<V2HashedStorageChangeSets>()?,
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
            self.tx.cursor_read::<V2HashedAccounts>()?,
            self.tx.cursor_read::<V2HashedAccountsHistory>()?,
            self.tx.cursor_read::<V2HashedAccountsHistory>()?,
            self.tx.cursor_dup_read::<V2HashedAccountChangeSets>()?,
            max_block_number,
            is_latest,
        ))
    }

    fn fetch_trie_updates(&self, block_number: u64) -> OpProofsStorageResult<BlockStateDiff> {
        self.fetch_trie_updates_inner(block_number)
    }
}

impl<TX: DbTxMut + DbTx + Send + Sync + Debug + 'static> OpProofsProviderRw
    for MdbxProofsProviderV2<TX>
{
    fn store_trie_updates(
        &self,
        block_ref: BlockWithParent,
        block_state_diff: BlockStateDiff,
    ) -> OpProofsStorageResult<WriteCounts> {
        self.validate_block_order(&block_ref)?;

        let mut collector = HistoryCollector::default();
        let mut cursors = WriteCursors::new(&self.tx)?;
        let counts = self.store_block_updates(
            block_ref.block.number,
            block_state_diff,
            &mut collector,
            &mut cursors,
        )?;
        drop(cursors);
        self.flush_collected_history(collector)?;

        self.set_latest_block_number_inner(block_ref.block.number, block_ref.block.hash)?;
        Ok(counts)
    }

    fn store_trie_updates_batch(
        &self,
        updates: Vec<(BlockWithParent, BlockStateDiff)>,
    ) -> OpProofsStorageResult<WriteCounts> {
        let mut total_counts = WriteCounts::default();
        let mut collector = HistoryCollector::default();
        let mut cursors = WriteCursors::new(&self.tx)?;

        // Track the latest hash in memory instead of reading/writing
        // V2ProofWindow per block (saves 2 cursor opens per block).
        let proof_window = self.get_proof_window_inner()?;
        let mut last_hash = proof_window.latest.hash;
        let mut last_written: Option<(BlockNumber, B256)> = None;

        for (block_ref, block_state_diff) in updates {
            let block_number = block_ref.block.number;

            if last_hash != block_ref.parent {
                return Err(OpProofsStorageError::OutOfOrder {
                    block_number,
                    parent_block_hash: block_ref.parent,
                    latest_block_hash: last_hash,
                });
            }

            let counts = self.store_block_updates(
                block_number,
                block_state_diff,
                &mut collector,
                &mut cursors,
            )?;

            last_hash = block_ref.block.hash;
            last_written = Some((block_number, block_ref.block.hash));
            total_counts += counts;
        }

        // Drop cursors before flush opens new ones for the history tables.
        drop(cursors);

        // Flush all history bitmap entries in one pass — each unique key is
        // seeked, decoded, and re-encoded exactly once regardless of how many
        // blocks in the batch touched it.
        self.flush_collected_history(collector)?;

        // Write V2ProofWindow once at the end instead of per-block.
        if let Some((number, hash)) = last_written {
            self.set_latest_block_number_inner(number, hash)?;
        }

        Ok(total_counts)
    }

    fn prune_earliest_state(
        &self,
        new_earliest_block_ref: BlockWithParent,
    ) -> OpProofsStorageResult<WriteCounts> {
        let target_block = new_earliest_block_ref.block.number;
        let proof_window = self.get_proof_window_inner()?;

        if proof_window.earliest.number >= target_block {
            return Ok(WriteCounts::default());
        }

        let range = (proof_window.earliest.number + 1)..=target_block;
        let mut counts = WriteCounts::default();

        // Phase A: scan and delete changesets, collecting affected keys for Phase B.
        let acct_trie_keys = self.prune_account_trie_changesets(&range, &mut counts)?;
        let stor_trie_keys = self.prune_storage_trie_changesets(&range, &mut counts)?;
        let acct_keys = self.prune_hashed_account_changesets(&range, &mut counts)?;
        let stor_keys = self.prune_hashed_storage_changesets(&range, &mut counts)?;

        // Phase B: remove pruned block numbers from history bitmaps.
        self.prune_account_trie_history(&range, &acct_trie_keys)?;
        self.prune_storage_trie_history(&range, &stor_trie_keys)?;
        self.prune_hashed_account_history(&range, &acct_keys)?;
        self.prune_hashed_storage_history(&range, &stor_keys)?;

        self.set_earliest_block_number_inner(target_block, new_earliest_block_ref.block.hash)?;

        Ok(counts)
    }

    fn unwind_history(&self, to: BlockWithParent) -> OpProofsStorageResult<()> {
        let proof_window = self.get_proof_window_inner()?;

        if to.block.number > proof_window.latest.number {
            return Ok(());
        }

        if to.block.number <= proof_window.earliest.number {
            return Err(OpProofsStorageError::UnwindBeyondEarliest {
                unwind_block_number: to.block.number,
                earliest_block_number: proof_window.earliest.number,
            });
        }

        let range = to.block.number..=proof_window.latest.number;

        // Single-scan: restore state, collect affected keys, delete changesets
        let acct_trie_keys = self.unwind_and_collect_account_trie(&range)?;
        let stor_trie_keys = self.unwind_and_collect_storage_trie(&range)?;
        let acct_keys = self.unwind_and_collect_hashed_accounts(&range)?;
        let stor_keys = self.unwind_and_collect_hashed_storages(&range)?;

        // Phase B: remove unwound block numbers from history bitmaps
        self.prune_account_trie_history(&range, &acct_trie_keys)?;
        self.prune_storage_trie_history(&range, &stor_trie_keys)?;
        self.prune_hashed_account_history(&range, &acct_keys)?;
        self.prune_hashed_storage_history(&range, &stor_keys)?;

        // Update latest block
        self.set_latest_block_number_inner(to.block.number.saturating_sub(1), to.parent)?;

        Ok(())
    }

    fn replace_updates(
        &self,
        latest_common_block: BlockNumHash,
        mut blocks_to_add: Vec<(BlockWithParent, BlockStateDiff)>,
    ) -> OpProofsStorageResult<()> {
        let proof_window = self.get_proof_window_inner()?;

        if latest_common_block.number < proof_window.earliest.number ||
            latest_common_block.number > proof_window.latest.number
        {
            return Err(OpProofsStorageError::ReorgBaseOutOfWindow {
                block_number: latest_common_block.number,
                earliest_block_number: proof_window.earliest.number,
                latest_block_number: proof_window.latest.number,
            });
        }

        blocks_to_add.sort_unstable_by_key(|(bwp, _)| bwp.block.number);

        // Phase 1: unwind to the latest common block, which is the new base of the proof window.
        {
            let range = (latest_common_block.number + 1)..=proof_window.latest.number;

            // Single-scan: restore state, collect affected keys, delete changesets
            let acct_trie_keys = self.unwind_and_collect_account_trie(&range)?;
            let stor_trie_keys = self.unwind_and_collect_storage_trie(&range)?;
            let acct_keys = self.unwind_and_collect_hashed_accounts(&range)?;
            let stor_keys = self.unwind_and_collect_hashed_storages(&range)?;

            // Phase B: remove old block numbers from history bitmaps
            self.prune_account_trie_history(&range, &acct_trie_keys)?;
            self.prune_storage_trie_history(&range, &stor_trie_keys)?;
            self.prune_hashed_account_history(&range, &acct_keys)?;
            self.prune_hashed_storage_history(&range, &stor_keys)?;
        }

        // Phase 2: add new blocks on top of the latest common block.
        // Re-add blocks using a shared collector + cursors, same as the batch
        // path, so history bitmap appends are batched and cursors are reused.
        // Track block ordering in memory to avoid per-block V2ProofWindow I/O.
        let mut last_hash = latest_common_block.hash;
        let mut last_written = latest_common_block;
        let mut collector = HistoryCollector::default();
        let mut cursors = WriteCursors::new(&self.tx)?;

        for (block_ref, diff) in blocks_to_add {
            let block_number = block_ref.block.number;

            if last_hash != block_ref.parent {
                return Err(OpProofsStorageError::OutOfOrder {
                    block_number,
                    parent_block_hash: block_ref.parent,
                    latest_block_hash: last_hash,
                });
            }

            self.store_block_updates(block_number, diff, &mut collector, &mut cursors)?;

            last_hash = block_ref.block.hash;
            last_written = NumHash::new(block_number, block_ref.block.hash);
        }

        drop(cursors);
        self.flush_collected_history(collector)?;

        self.set_latest_block_number_inner(last_written.number, last_written.hash)?;

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

impl<TX: DbTxMut + DbTx + Send + Sync + Debug + 'static> OpProofsInitProvider
    for MdbxProofsProviderV2<TX>
{
    fn initial_state_anchor(&self) -> OpProofsStorageResult<InitialStateAnchor> {
        let Some(block) = self.get_initial_state_anchor_inner()? else {
            return Ok(InitialStateAnchor::default());
        };

        let completed = self.get_block_number_hash_inner(ProofWindowKey::EarliestBlock)?.is_some();

        // Scan the last entry in each current-state table to determine resume
        // keys. This allows multi-step initialization: if the process is
        // interrupted, the next run picks up where it left off.
        let latest_hashed_account_key =
            self.tx.cursor_read::<V2HashedAccounts>()?.last()?.map(|(k, _)| k);

        let latest_hashed_storage_key = self
            .tx
            .cursor_read::<V2HashedStorages>()?
            .last()?
            .map(|(addr, entry)| HashedStorageKey::new(addr, entry.key));

        let latest_account_trie_key =
            self.tx.cursor_read::<V2AccountsTrie>()?.last()?.map(|(k, _)| k);

        let latest_storage_trie_key = self
            .tx
            .cursor_read::<V2StoragesTrie>()?
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
        let mut cur = self.tx.cursor_write::<V2ProofWindow>()?;
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

        let mut cursor = self.tx.cursor_write::<V2AccountsTrie>()?;
        for (nibbles, maybe_node) in account_nodes {
            if let Some(node) = maybe_node {
                cursor.upsert(StoredNibbles(nibbles), &node)?;
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

        let mut cursor = self.tx.cursor_dup_write::<V2StoragesTrie>()?;
        for (nibbles, maybe_node) in storage_nodes {
            if let Some(node) = maybe_node {
                cursor.append_dup(
                    hashed_address,
                    StorageTrieEntry { nibbles: StoredNibblesSubKey(nibbles), node },
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

        let mut cursor = self.tx.cursor_write::<V2HashedAccounts>()?;
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

        let mut cursor = self.tx.cursor_dup_write::<V2HashedStorages>()?;
        for (storage_key, value) in storages {
            cursor.append_dup(hashed_address, StorageEntry { key: storage_key, value })?;
        }
        Ok(())
    }

    fn commit_initial_state(&self) -> OpProofsStorageResult<BlockNumHash> {
        let anchor =
            self.get_initial_state_anchor_inner()?.ok_or(OpProofsStorageError::NoBlocksFound)?;
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
        Database, DatabaseEnv,
        cursor::DbDupCursorRO,
        mdbx::{DatabaseArguments, init_db_for},
        transaction::DbTx,
    };
    use reth_trie::{
        HashedStorage,
        updates::{StorageTrieUpdates, TrieUpdates},
    };
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
        let mut cur = tx.cursor_read::<V2HashedAccounts>().expect("cursor");
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
        let mut cur = tx.cursor_dup_read::<V2HashedStorages>().expect("cursor");
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
        let mut cur = tx.cursor_read::<V2AccountsTrie>().expect("cursor");
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
        let mut cur = tx.cursor_dup_read::<V2StoragesTrie>().expect("cursor");
        let entry =
            cur.seek_by_key_subkey(addr, StoredNibblesSubKey(path)).expect("seek").expect("exists");
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
            let mut cur = tx.cursor_read::<V2HashedAccounts>().expect("cursor");
            let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
            assert_eq!(acc.nonce, 2, "current state should have updated nonce");
        }

        // Verify changeset has the old account
        {
            let tx = db.tx().expect("ro");
            let mut cur = tx.cursor_dup_read::<V2HashedAccountChangeSets>().expect("cursor");
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
            let mut cur = tx.cursor_read::<V2HashedAccounts>().expect("cursor");
            let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
            assert_eq!(acc.nonce, 1);
        }

        // Unwind block 1
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            let unwind_to =
                BlockWithParent::new(B256::ZERO, NumHash::new(1, B256::repeat_byte(0x01)));
            provider.unwind_history(unwind_to).expect("unwind");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Verify v0 is restored
        {
            let tx = db.tx().expect("ro");
            let mut cur = tx.cursor_read::<V2HashedAccounts>().expect("cursor");
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
            provider.store_hashed_accounts(vec![(addr, Some(Account::default()))]).expect("init");
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
            post_state
                .accounts
                .insert(addr, Some(Account { nonce: block_num, ..Default::default() }));

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
            let mut cur = tx.cursor_read::<V2HashedAccountsHistory>().expect("cursor");
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
            provider.store_hashed_accounts(vec![(addr, Some(Account::default()))]).expect("init");
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
            post_state
                .accounts
                .insert(addr, Some(Account { nonce: block_num, ..Default::default() }));
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
            let mut cur = tx.cursor_read::<V2HashedAccountChangeSets>().expect("cursor");
            // Block 1 should be gone
            assert!(
                cur.seek_exact(1u64).expect("seek").is_none(),
                "block 1 changeset should be pruned"
            );
            // Block 2 should be gone
            assert!(
                cur.seek_exact(2u64).expect("seek").is_none(),
                "block 2 changeset should be pruned"
            );
            // Block 3 should still exist
            assert!(
                cur.seek_exact(3u64).expect("seek").is_some(),
                "block 3 changeset should remain"
            );
        }

        // Current state should still be at block 3
        {
            let tx = db.tx().expect("ro");
            let mut cur = tx.cursor_read::<V2HashedAccounts>().expect("cursor");
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
        provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
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
            provider.store_hashed_accounts(vec![(addr1, Some(acc1_old))]).expect("init accounts");
            provider.store_hashed_storages(addr1, vec![(slot1, val1_old)]).expect("init storage");
            provider
                .store_account_branches(vec![
                    (path1, Some(node1_old.clone())),
                    (removed_path, Some(removed_node_old.clone())),
                ])
                .expect("init account trie");
            provider
                .store_storage_branches(addr1, vec![(storage_path1, Some(snode1_old))])
                .expect("init storage trie");
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
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
        let mut acc_cur = tx.cursor_read::<V2HashedAccounts>().expect("cursor");
        let (_, acc) = acc_cur.seek_exact(addr1).expect("seek").expect("exists");
        assert_eq!(acc.nonce, acc1_new.nonce);
        assert!(acc_cur.seek_exact(addr2).expect("seek").is_none(), "addr2 was never created");

        // Storage: addr1/slot1 should have new value
        let mut stor_cur = tx.cursor_dup_read::<V2HashedStorages>().expect("cursor");
        let entry = stor_cur.seek_by_key_subkey(addr1, slot1).expect("seek").expect("exists");
        assert_eq!(entry.value, val1_new);

        // Account trie: path1 new, path2 new, removed_path gone
        let mut trie_cur = tx.cursor_read::<V2AccountsTrie>().expect("cursor");
        let (_, n) = trie_cur.seek_exact(StoredNibbles(path1)).expect("seek").expect("exists");
        assert_eq!(n, node1_new);
        let (_, n2) = trie_cur.seek_exact(StoredNibbles(path2)).expect("seek").expect("exists");
        assert_eq!(n2, node2_new);
        assert!(
            trie_cur.seek_exact(StoredNibbles(removed_path)).expect("seek").is_none(),
            "removed path should be gone"
        );

        // Storage trie: addr1/storage_path1 should have new node
        let mut strie_cur = tx.cursor_dup_read::<V2StoragesTrie>().expect("cursor");
        let e = strie_cur
            .seek_by_key_subkey(addr1, StoredNibblesSubKey(storage_path1))
            .expect("seek")
            .expect("exists");
        assert_eq!(e.node, snode1_new);

        // Verify account changeset has old values
        let mut cs_cur = tx.cursor_read::<V2HashedAccountChangeSets>().expect("cursor");
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
        let mut tcs_cur = tx.cursor_read::<V2AccountTrieChangeSets>().expect("cursor");
        let mut tentries = Vec::new();
        let mut walker = tcs_cur.walk(Some(42u64)).expect("walk");
        while let Some(Ok((bn, entry))) = walker.next() {
            if bn != 42 {
                break;
            }
            tentries.push(entry);
        }
        assert!(tentries.iter().any(|e| e.nibbles.0 == path1 && e.node == Some(node1_old.clone())));
        assert!(
            tentries
                .iter()
                .any(|e| e.nibbles.0 == removed_path && e.node == Some(removed_node_old.clone()))
        );
        assert!(tentries.iter().any(|e| e.nibbles.0 == path2 && e.node.is_none()));

        // Verify V2ProofWindow latest
        let mut pw_cur = tx.cursor_read::<V2ProofWindow>().expect("cursor");
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
        let mut cur1 = tx.cursor_read::<V2HashedAccountChangeSets>().expect("cursor");
        assert!(cur1.first().expect("first").is_none(), "Account changesets should be empty");
        let mut cur2 = tx.cursor_read::<V2AccountTrieChangeSets>().expect("cursor");
        assert!(cur2.first().expect("first").is_none(), "Account trie changesets should be empty");

        // V2ProofWindow should be updated
        let mut pw_cur = tx.cursor_read::<V2ProofWindow>().expect("cursor");
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
        let mut cur = tx.cursor_read::<V2HashedAccounts>().expect("cursor");
        let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
        assert_eq!(acc.nonce, 20);

        // Changeset at block 1 should have old nonce (0)
        let mut cs = tx.cursor_read::<V2HashedAccountChangeSets>().expect("cursor");
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
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
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
        let mut cur = tx.cursor_read::<V2AccountsTrie>().expect("cursor");
        assert!(
            cur.seek_exact(StoredNibbles(acc_path)).expect("seek").is_none(),
            "node should be removed from current state"
        );

        // Changeset should have the old node
        let mut cs = tx.cursor_read::<V2AccountTrieChangeSets>().expect("cursor");
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
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
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
        let mut cur = tx.cursor_dup_read::<V2StoragesTrie>().expect("cursor");
        let result = cur
            .seek_by_key_subkey(addr, StoredNibblesSubKey(st_path))
            .expect("seek")
            .filter(|e| e.nibbles.0 == st_path);
        assert!(result.is_none(), "node should be removed from current state");

        // Changeset should have the old node
        let mut cs = tx.cursor_read::<V2StorageTrieChangeSets>().expect("cursor");
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
                .store_storage_branches(addr_wiped, vec![(p1, Some(n1)), (p2, Some(n2))])
                .expect("seed");
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
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
        let mut cur = tx.cursor_dup_read::<V2StoragesTrie>().expect("cursor");
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
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
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
        let mut cur = tx.cursor_dup_read::<V2HashedStorages>().expect("cursor");
        assert!(
            cur.seek_exact(addr).expect("seek").is_none(),
            "wiped storage should have no entries in current state"
        );

        // Changeset should have old values
        let mut cs = tx.cursor_read::<V2HashedStorageChangeSets>().expect("cursor");
        let start = BlockNumberHashedAddress((42, addr));
        let mut old_values = Vec::new();
        let mut walker = cs.walk(Some(start)).expect("walk");
        while let Some(Ok((key, entry))) = walker.next() {
            if key.0.0 != 42 || key.0.1 != addr {
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
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
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
        let mut cur = tx.cursor_dup_read::<V2HashedStorages>().expect("cursor");
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
            provider.store_hashed_accounts(vec![(addr1, Some(acc1_old))]).expect("init accounts");
            provider.store_hashed_storages(addr1, vec![(slot1, val1_old)]).expect("init storage");
            provider.store_account_branches(vec![(path1, Some(node1_old))]).expect("init trie");
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Build diff and store
        let mut trie_updates = TrieUpdates::default();
        trie_updates.account_nodes.insert(path1, node1_new);

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
        assert!(
            got.sorted_post_state.accounts.iter().any(|(a, v)| *a == addr1 && v == &Some(acc1_new))
        );

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
        // Don't initialize — pruning uninitialized store returns NoBlocksFound.

        let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
        let target = make_block_ref(5, B256::repeat_byte(0x05), B256::ZERO);
        let result = provider.prune_earliest_state(target);
        assert!(
            matches!(result, Err(OpProofsStorageError::NoBlocksFound)),
            "expected NoBlocksFound, got {result:?}"
        );
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
        let mut cur = tx.cursor_read::<V2HashedAccounts>().expect("cursor");
        let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
        assert_eq!(acc.nonce, 2, "current state should still have latest value");

        // Changesets for blocks 1 and 2 should be gone
        let mut cs = tx.cursor_read::<V2HashedAccountChangeSets>().expect("cursor");
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
            provider.store_hashed_accounts(vec![(addr, Some(Account::default()))]).expect("init");
            provider.store_hashed_storages(addr, vec![(slot, U256::from(100u64))]).expect("init");
            provider.store_account_branches(vec![(path, Some(node_old))]).expect("init");
            provider
                .store_storage_branches(addr, vec![(storage_path, Some(snode_old))])
                .expect("init");
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
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
        let mut acc_cur = tx.cursor_read::<V2HashedAccounts>().expect("cursor");
        let (_, acc) = acc_cur.seek_exact(addr).expect("seek").expect("exists");
        assert_eq!(acc.nonce, 2);

        // Changesets should be gone for blocks 1 and 2
        let mut cs = tx.cursor_read::<V2HashedAccountChangeSets>().expect("cursor");
        assert!(cs.seek_exact(1u64).expect("seek").is_none());
        assert!(cs.seek_exact(2u64).expect("seek").is_none());

        let mut tcs = tx.cursor_read::<V2AccountTrieChangeSets>().expect("cursor");
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
            let prune_ref = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));
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
            let mut cur = tx.cursor_read::<V2HashedAccounts>().expect("cursor");
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
            let mut cur = tx.cursor_read::<V2HashedAccounts>().expect("cursor");
            let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
            assert_eq!(acc.nonce, 400);
        }

        // Verify: changesets exist for blocks 1, 2, 3', 4'
        {
            let tx = db.tx().expect("ro");
            let mut cs = tx.cursor_read::<V2HashedAccountChangeSets>().expect("cursor");
            assert!(cs.seek_exact(1u64).expect("seek").is_some(), "block 1 changeset");
            assert!(cs.seek_exact(2u64).expect("seek").is_some(), "block 2 changeset");
            assert!(cs.seek_exact(3u64).expect("seek").is_some(), "block 3' changeset");
            assert!(cs.seek_exact(4u64).expect("seek").is_some(), "block 4' changeset");
        }
    }

    #[test]
    fn test_replace_updates_beyond_earliest_returns_error() {
        let db = setup_db();
        let addr = B256::from([0xCC; 32]);
        init_state(&db, vec![(addr, Some(Account::default()))]);

        // Build chain: 1 -> 2 -> 3, then prune to earliest = 2.
        let b1 = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
        let b2 = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));
        let b3 = make_block_ref(3, B256::repeat_byte(0x03), B256::repeat_byte(0x02));

        store_block(&db, b1, make_nonce_diff(addr, 10));
        store_block(&db, b2, make_nonce_diff(addr, 20));
        store_block(&db, b3, make_nonce_diff(addr, 30));

        // Move earliest forward to block 2.
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider
                .prune_earliest_state(make_block_ref(
                    2,
                    B256::repeat_byte(0x02),
                    B256::repeat_byte(0x01),
                ))
                .expect("prune");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Attempt to replace_updates with a common block before earliest (block 1).
        let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
        let res = provider.replace_updates(BlockNumHash::new(1, B256::repeat_byte(0x01)), vec![]);
        assert!(
            matches!(res, Err(OpProofsStorageError::ReorgBaseOutOfWindow { .. })),
            "expected ReorgBaseOutOfWindow, got {res:?}"
        );
    }

    #[test]
    fn test_replace_updates_ahead_of_latest_returns_error() {
        let db = setup_db();
        let addr = B256::from([0xDD; 32]);
        init_state(&db, vec![(addr, Some(Account::default()))]);

        // Build chain: 1 -> 2. Latest = 2.
        let b1 = make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO);
        let b2 = make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01));

        store_block(&db, b1, make_nonce_diff(addr, 10));
        store_block(&db, b2, make_nonce_diff(addr, 20));

        // Attempt to replace_updates with a common block beyond latest (block 5).
        let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
        let res = provider.replace_updates(BlockNumHash::new(5, B256::repeat_byte(0x05)), vec![]);
        assert!(
            matches!(res, Err(OpProofsStorageError::ReorgBaseOutOfWindow { .. })),
            "expected ReorgBaseOutOfWindow, got {res:?}"
        );
    }

    // ========================== Unwind tests ==========================

    #[test]
    fn test_unwind_history_to_earliest() {
        let db = setup_db();
        let addr = B256::from([0xBB; 32]);

        // Initialize and set earliest at block 1
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider.store_hashed_accounts(vec![(addr, Some(Account::default()))]).expect("init");
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
        let unwind_to =
            BlockWithParent::new(B256::repeat_byte(0x01), NumHash::new(1, B256::repeat_byte(0x01)));
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
            provider.store_hashed_accounts(vec![(addr, Some(Account::default()))]).expect("init");
            provider.store_hashed_storages(addr, vec![(slot, U256::from(100u64))]).expect("init");
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Block 1: update both account and storage
        {
            let mut post_state = HashedPostState::default();
            post_state.accounts.insert(addr, Some(Account { nonce: 1, ..Default::default() }));
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
            post_state.accounts.insert(addr, Some(Account { nonce: 2, ..Default::default() }));
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
        let mut acc_cur = tx.cursor_read::<V2HashedAccounts>().expect("cursor");
        let (_, acc) = acc_cur.seek_exact(addr).expect("seek").expect("exists");
        assert_eq!(acc.nonce, 1, "account should be restored to block 1 state");

        let mut stor_cur = tx.cursor_dup_read::<V2HashedStorages>().expect("cursor");
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
            provider.store_account_branches(vec![(path1, Some(node1.clone()))]).expect("init");
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
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
        let mut cur = tx.cursor_read::<V2AccountsTrie>().expect("cursor");
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
            provider.store_hashed_accounts(vec![(addr1, Some(acc1))]).expect("init");
            provider
                .store_hashed_storages(addr1, vec![(slot1, U256::from(1111u64))])
                .expect("init");
            provider.store_account_branches(vec![(path1, Some(node1))]).expect("init");
            provider
                .store_storage_branches(addr1, vec![(storage_path1, Some(snode1))])
                .expect("init");
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
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
            post_state.accounts.insert(addr1, Some(Account { nonce: 10, ..Default::default() }));
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
            post_state.accounts.insert(addr2, Some(Account { nonce: 20, ..Default::default() }));

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
        let mut acc_cur = tx.cursor_read::<V2HashedAccounts>().expect("cursor");
        let (_, acc) = acc_cur.seek_exact(addr1).expect("seek").expect("exists");
        assert_eq!(acc.nonce, 10, "addr1 should have block 1 state");
        // addr2 should not exist (was added in block 2, unwound)
        assert!(acc_cur.seek_exact(addr2).expect("seek").is_none(), "addr2 should be removed");

        // Verify trie: path1 should have block 1 value
        let mut trie_cur = tx.cursor_read::<V2AccountsTrie>().expect("cursor");
        assert!(trie_cur.seek_exact(StoredNibbles(path1)).expect("seek").is_some());
        // path2 should not exist (added in block 2, unwound)
        assert!(
            trie_cur.seek_exact(StoredNibbles(path2)).expect("seek").is_none(),
            "path2 should be removed"
        );

        // Verify storage: should have block 1 value
        let mut stor_cur = tx.cursor_dup_read::<V2HashedStorages>().expect("cursor");
        let entry = stor_cur.seek_by_key_subkey(addr1, slot1).expect("seek").expect("exists");
        assert_eq!(entry.value, U256::from(2222u64), "storage should have block 1 value");

        // Verify changesets for blocks 2+ are gone
        let mut cs = tx.cursor_read::<V2HashedAccountChangeSets>().expect("cursor");
        assert!(cs.seek_exact(1u64).expect("seek").is_some(), "block 1 changeset should remain");
        assert!(
            cs.seek_exact(2u64).expect("seek").is_none(),
            "block 2 changeset should be removed"
        );

        // Verify V2ProofWindow latest
        let mut pw_cur = tx.cursor_read::<V2ProofWindow>().expect("cursor");
        let (_, val) =
            pw_cur.seek_exact(ProofWindowKey::LatestBlock).expect("seek").expect("exists");
        assert_eq!(val.number(), 1);
    }

    #[test]
    fn test_unwind_history_empty_chain() {
        let db = setup_db();

        // No blocks stored — uninitialized proof window returns NoBlocksFound.
        let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
        let unwind_to = BlockWithParent::new(B256::ZERO, NumHash::new(0, B256::ZERO));
        let result = provider.unwind_history(unwind_to);
        assert!(
            matches!(result, Err(OpProofsStorageError::NoBlocksFound)),
            "expected NoBlocksFound, got {result:?}"
        );
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
        let mut cur = tx.cursor_read::<V2HashedAccounts>().expect("cursor");
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
        let mut cur = tx.cursor_read::<V2HashedAccounts>().expect("cursor");
        let (_, acc) = cur.seek_exact(addr).expect("seek").expect("exists");
        assert_eq!(acc.nonce, 30, "state should be unchanged");

        let mut pw_cur = tx.cursor_read::<V2ProofWindow>().expect("cursor");
        let (_, val) =
            pw_cur.seek_exact(ProofWindowKey::LatestBlock).expect("seek").expect("exists");
        assert_eq!(val.number(), 3, "latest should be unchanged");
    }

    /// Helper: count the total number of duplicate entries for a given primary key
    /// in the `V2HashedStorages` `DupSort` table.
    fn count_hashed_storage_entries(db: &DatabaseEnv, addr: B256) -> usize {
        let tx = db.tx().expect("ro");
        let mut cur = tx.cursor_dup_read::<V2HashedStorages>().expect("cursor");
        let mut count = 0;
        if cur.seek_by_key_subkey(addr, B256::ZERO).expect("seek").is_some() {
            count += 1;
            while cur.next_dup_val().expect("next").is_some() {
                count += 1;
            }
        }
        count
    }

    /// Helper: collect all (slot, value) pairs for an address from `V2HashedStorages`.
    fn collect_hashed_storage_slots(db: &DatabaseEnv, addr: B256) -> Vec<(B256, U256)> {
        let tx = db.tx().expect("ro");
        let mut cur = tx.cursor_dup_read::<V2HashedStorages>().expect("cursor");
        let mut entries = Vec::new();
        if let Some(entry) = cur.seek_by_key_subkey(addr, B256::ZERO).expect("seek") {
            entries.push((entry.key, entry.value));
            while let Some(entry) = cur.next_dup_val().expect("next") {
                entries.push((entry.key, entry.value));
            }
        }
        entries
    }

    /// Regression: updating the same slot across multiple blocks must NOT create
    /// duplicate entries. Each (address, slot) pair should appear exactly once in
    /// the current-state table.
    #[test]
    fn hashed_storages_no_duplicate_entries_after_multi_block_update() {
        let db = setup_db();

        let addr = B256::from([0xDE; 32]);
        let slot = B256::from([0xAB; 32]);

        // Initialize with one storage slot
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider.store_hashed_storages(addr, vec![(slot, U256::from(100u64))]).expect("seed");
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Verify initial state: exactly 1 entry
        assert_eq!(count_hashed_storage_entries(&db, addr), 1, "initial: exactly 1 entry");

        // Store 5 blocks, each updating the same slot to a new value
        let mut parent = B256::ZERO;
        for block_num in 1u64..=5 {
            let hash = B256::repeat_byte(block_num as u8);
            let mut post_state = HashedPostState::default();
            let mut storage = HashedStorage::default();
            storage.storage.insert(slot, U256::from(block_num * 1000));
            post_state.storages.insert(addr, storage);

            let diff = BlockStateDiff {
                sorted_trie_updates: TrieUpdates::default().into_sorted(),
                sorted_post_state: post_state.into_sorted(),
            };
            store_block(&db, make_block_ref(block_num, hash, parent), diff);
            parent = hash;

            // After each block: still exactly 1 entry
            assert_eq!(
                count_hashed_storage_entries(&db, addr),
                1,
                "block {block_num}: must still be exactly 1 entry, no duplicates"
            );
        }

        // Verify final value is correct
        let slots = collect_hashed_storage_slots(&db, addr);
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0], (slot, U256::from(5000u64)));
    }

    /// Regression: multiple slots for the same address should each appear exactly
    /// once after updates across blocks.
    #[test]
    fn hashed_storages_no_duplicates_multiple_slots() {
        let db = setup_db();

        let addr = B256::from([0xCC; 32]);
        let slot_a = B256::from([0x01; 32]);
        let slot_b = B256::from([0x02; 32]);
        let slot_c = B256::from([0x03; 32]);

        // Initialize with 2 slots
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider
                .store_hashed_storages(
                    addr,
                    vec![(slot_a, U256::from(10u64)), (slot_b, U256::from(20u64))],
                )
                .expect("seed");
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        assert_eq!(count_hashed_storage_entries(&db, addr), 2, "initial: 2 entries");

        // Block 1: update slot_a, add slot_c
        {
            let mut post_state = HashedPostState::default();
            let mut storage = HashedStorage::default();
            storage.storage.insert(slot_a, U256::from(11u64));
            storage.storage.insert(slot_c, U256::from(30u64));
            post_state.storages.insert(addr, storage);

            let diff = BlockStateDiff {
                sorted_trie_updates: TrieUpdates::default().into_sorted(),
                sorted_post_state: post_state.into_sorted(),
            };
            store_block(&db, make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO), diff);
        }

        // Should be 3 entries: slot_a (updated), slot_b (untouched), slot_c (new)
        assert_eq!(count_hashed_storage_entries(&db, addr), 3, "block 1: exactly 3 entries");

        // Block 2: update all 3 slots
        {
            let mut post_state = HashedPostState::default();
            let mut storage = HashedStorage::default();
            storage.storage.insert(slot_a, U256::from(12u64));
            storage.storage.insert(slot_b, U256::from(22u64));
            storage.storage.insert(slot_c, U256::from(32u64));
            post_state.storages.insert(addr, storage);

            let diff = BlockStateDiff {
                sorted_trie_updates: TrieUpdates::default().into_sorted(),
                sorted_post_state: post_state.into_sorted(),
            };
            store_block(
                &db,
                make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01)),
                diff,
            );
        }

        // Still 3 entries — no duplicates
        assert_eq!(count_hashed_storage_entries(&db, addr), 3, "block 2: exactly 3, no dupes");

        // Block 3: delete slot_b (set to zero)
        {
            let mut post_state = HashedPostState::default();
            let mut storage = HashedStorage::default();
            storage.storage.insert(slot_b, U256::ZERO);
            post_state.storages.insert(addr, storage);

            let diff = BlockStateDiff {
                sorted_trie_updates: TrieUpdates::default().into_sorted(),
                sorted_post_state: post_state.into_sorted(),
            };
            store_block(
                &db,
                make_block_ref(3, B256::repeat_byte(0x03), B256::repeat_byte(0x02)),
                diff,
            );
        }

        // 2 entries: slot_a and slot_c remain, slot_b deleted
        let slots = collect_hashed_storage_slots(&db, addr);
        assert_eq!(slots.len(), 2, "block 3: slot_b deleted, 2 remain");
        assert!(slots.iter().all(|(k, _)| *k != slot_b), "slot_b should be gone");
    }

    /// Regression: wipe followed by re-add in same block must leave exactly the
    /// new slots, no ghosts from pre-wipe state.
    #[test]
    fn hashed_storages_wipe_then_readd_no_duplicates() {
        let db = setup_db();

        let addr = B256::from([0xEE; 32]);
        let old_slot = B256::from([0x01; 32]);
        let new_slot = B256::from([0x02; 32]);

        // Initialize with old_slot
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider
                .store_hashed_storages(addr, vec![(old_slot, U256::from(999u64))])
                .expect("seed");
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        assert_eq!(count_hashed_storage_entries(&db, addr), 1);

        // Block 1: wipe + write new_slot
        {
            let mut post_state = HashedPostState::default();
            let mut storage = HashedStorage::new(true); // wiped = true
            storage.storage.insert(new_slot, U256::from(42u64));
            post_state.storages.insert(addr, storage);

            let diff = BlockStateDiff {
                sorted_trie_updates: TrieUpdates::default().into_sorted(),
                sorted_post_state: post_state.into_sorted(),
            };
            store_block(&db, make_block_ref(1, B256::repeat_byte(0x01), B256::ZERO), diff);
        }

        // Exactly 1 entry: only new_slot, old_slot wiped
        let slots = collect_hashed_storage_slots(&db, addr);
        assert_eq!(slots.len(), 1, "after wipe+add: exactly 1 entry");
        assert_eq!(slots[0], (new_slot, U256::from(42u64)));

        // Block 2: wipe + re-add the same new_slot with different value
        {
            let mut post_state = HashedPostState::default();
            let mut storage = HashedStorage::new(true);
            storage.storage.insert(new_slot, U256::from(84u64));
            post_state.storages.insert(addr, storage);

            let diff = BlockStateDiff {
                sorted_trie_updates: TrieUpdates::default().into_sorted(),
                sorted_post_state: post_state.into_sorted(),
            };
            store_block(
                &db,
                make_block_ref(2, B256::repeat_byte(0x02), B256::repeat_byte(0x01)),
                diff,
            );
        }

        // Still exactly 1 entry
        let slots = collect_hashed_storage_slots(&db, addr);
        assert_eq!(slots.len(), 1, "after second wipe+add: exactly 1 entry");
        assert_eq!(slots[0], (new_slot, U256::from(84u64)));
    }

    /// Regression: batch store (`store_trie_updates_batch`) updating the same slot
    /// across multiple blocks in one batch must not leak duplicates.
    #[test]
    fn hashed_storages_batch_no_duplicates() {
        let db = setup_db();

        let addr = B256::from([0xBB; 32]);
        let slot = B256::from([0xAA; 32]);

        // Initialize
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider.store_hashed_storages(addr, vec![(slot, U256::from(1u64))]).expect("seed");
            provider.set_initial_state_anchor(BlockNumHash::new(0, B256::ZERO)).expect("anchor");
            provider.commit_initial_state().expect("commit init");
            OpProofsInitProvider::commit(provider).expect("commit");
        }

        // Build 3 blocks in a batch, each updating the same slot
        let blocks: Vec<(BlockWithParent, BlockStateDiff)> = (1u64..=3)
            .map(|n| {
                let mut post_state = HashedPostState::default();
                let mut storage = HashedStorage::default();
                storage.storage.insert(slot, U256::from(n * 100));
                post_state.storages.insert(addr, storage);

                let diff = BlockStateDiff {
                    sorted_trie_updates: TrieUpdates::default().into_sorted(),
                    sorted_post_state: post_state.into_sorted(),
                };
                let parent = if n == 1 { B256::ZERO } else { B256::repeat_byte((n - 1) as u8) };
                (make_block_ref(n, B256::repeat_byte(n as u8), parent), diff)
            })
            .collect();

        // Store as batch
        {
            let provider = MdbxProofsProviderV2::new(db.tx_mut().expect("rw"));
            provider.store_trie_updates_batch(blocks).expect("batch store");
            OpProofsProviderRw::commit(provider).expect("commit");
        }

        // Exactly 1 entry with final value
        let slots = collect_hashed_storage_slots(&db, addr);
        assert_eq!(slots.len(), 1, "batch: exactly 1 entry, no duplicates");
        assert_eq!(slots[0], (slot, U256::from(300u64)));
    }
}
