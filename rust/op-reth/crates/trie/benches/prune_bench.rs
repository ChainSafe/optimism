//! Pruning performance benchmark: V1 (versioned DupSort) vs V2 (3-table pattern).
//!
//! Measures `prune_earliest_state` under varying dataset sizes:
//!   - Scales: 100, 500, 2000 blocks
//!   - Each block touches ~50 accounts, ~200 storage slots, ~30 account trie nodes, ~100 storage
//!     trie nodes
//!
//! Run:  cargo bench -p reth-optimism-trie --bench prune_bench

#![allow(missing_docs)]

use alloy_eips::{eip1898::BlockWithParent, NumHash};
use alloy_primitives::{B256, U256};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use reth_optimism_trie::{
    api::{OpProofsInitProvider, OpProofsProviderRw, OpProofsStore, WriteCounts},
    BlockStateDiff, MdbxProofsStorage, MdbxProofsStorageV2,
};
use reth_primitives_traits::Account;
use reth_trie_common::{
    updates::{StorageTrieUpdatesSorted, TrieUpdatesSorted},
    BranchNodeCompact, HashedPostStateSorted, HashedStorageSorted, Nibbles,
};
use std::collections::HashMap;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Constants – tune these to simulate realistic block sizes
// ---------------------------------------------------------------------------

/// Number of distinct accounts touched per block
const ACCOUNTS_PER_BLOCK: usize = 50;
/// Number of storage slots written per block (spread across accounts)
const STORAGE_SLOTS_PER_BLOCK: usize = 200;
/// Number of account trie branch nodes updated per block
const ACCOUNT_TRIE_NODES_PER_BLOCK: usize = 30;
/// Number of storage trie branch nodes updated per block (spread across accounts)
const STORAGE_TRIE_NODES_PER_BLOCK: usize = 100;

// ---------------------------------------------------------------------------
// Data generators
// ---------------------------------------------------------------------------

/// Generate a deterministic B256 from an index
fn addr(i: u64) -> B256 {
    let mut buf = [0u8; 32];
    buf[24..].copy_from_slice(&i.to_be_bytes());
    B256::from(buf)
}

/// Generate a deterministic nibbles path from two indices.
/// Nibble values must be in range 0..16.
fn nibbles_path(a: u64, b: u64) -> Nibbles {
    let mut nibs = Vec::with_capacity(8);
    // Extract low nibbles from each byte of `a` and `b`
    let a_bytes = a.to_be_bytes();
    let b_bytes = b.to_be_bytes();
    for &byte in &a_bytes[6..] {
        nibs.push(byte & 0x0f);
    }
    for &byte in &b_bytes[6..] {
        nibs.push(byte & 0x0f);
    }
    Nibbles::from_nibbles(nibs)
}

fn sample_account(nonce: u64) -> Account {
    Account {
        nonce,
        balance: U256::from(1000u64),
        ..Default::default()
    }
}

fn sample_node(seed: u64) -> BranchNodeCompact {
    BranchNodeCompact::new(
        (seed as u16) | 0b11,
        0,
        0,
        vec![],
        Some(B256::from(addr(seed))),
    )
}

/// Per-block entry counts for benchmark data generation.
#[derive(Clone, Copy)]
struct DiffConfig {
    accounts: usize,
    storage_slots: usize,
    account_trie_nodes: usize,
    storage_trie_nodes: usize,
}

impl DiffConfig {
    /// Default config matching the original constants.
    const DEFAULT: Self = Self {
        accounts: ACCOUNTS_PER_BLOCK,
        storage_slots: STORAGE_SLOTS_PER_BLOCK,
        account_trie_nodes: ACCOUNT_TRIE_NODES_PER_BLOCK,
        storage_trie_nodes: STORAGE_TRIE_NODES_PER_BLOCK,
    };

    /// Base mainnet average per-block counts.
    const BASE: Self = Self {
        accounts: 342,
        storage_slots: 1_660,
        account_trie_nodes: 1_649,
        storage_trie_nodes: 7_340,
    };
}

/// Build a `BlockStateDiff` for block `block_number` using the given config.
fn make_diff_with_config(block_number: u64, cfg: DiffConfig) -> BlockStateDiff {
    // ---- accounts ----
    let mut accounts = Vec::with_capacity(cfg.accounts);
    let addr_space = std::cmp::max(cfg.accounts as u64 * 10, 500);
    for i in 0..cfg.accounts as u64 {
        let idx = (block_number * 7 + i * 13) % addr_space;
        accounts.push((addr(idx), Some(sample_account(block_number + i))));
    }
    accounts.sort_by_key(|(k, _)| *k);
    accounts.dedup_by_key(|(k, _)| *k);

    // ---- storage slots (spread over accounts) ----
    let num_storage_accounts = std::cmp::max(cfg.accounts / 5, 10) as u64;
    let slot_space = std::cmp::max(cfg.storage_slots as u64 * 5, 1000);
    let mut storages_map: HashMap<B256, Vec<(B256, U256)>> = HashMap::new();
    for i in 0..cfg.storage_slots as u64 {
        let acct_idx = (block_number * 3 + i) % num_storage_accounts;
        let acct_addr = addr(acct_idx);
        let slot_idx = (block_number * 11 + i * 17) % slot_space;
        storages_map
            .entry(acct_addr)
            .or_default()
            .push((addr(slot_idx), U256::from(block_number + i)));
    }
    let mut storages: alloy_primitives::map::B256Map<HashedStorageSorted> =
        alloy_primitives::map::B256Map::default();
    for (acct, mut slots) in storages_map {
        slots.sort_by_key(|(k, _)| *k);
        slots.dedup_by_key(|(k, _)| *k);
        storages.insert(
            acct,
            HashedStorageSorted {
                storage_slots: slots,
                wiped: false,
            },
        );
    }

    // ---- account trie nodes ----
    let trie_space = std::cmp::max(cfg.account_trie_nodes as u64 * 5, 200);
    let mut account_nodes = Vec::with_capacity(cfg.account_trie_nodes);
    for i in 0..cfg.account_trie_nodes as u64 {
        let path_idx = (block_number * 5 + i * 19) % trie_space;
        account_nodes.push((nibbles_path(path_idx, 0), Some(sample_node(path_idx))));
    }
    account_nodes.sort_by_key(|(k, _)| k.clone());
    account_nodes.dedup_by_key(|(k, _)| k.clone());

    // ---- storage trie nodes (spread over accounts) ----
    let num_strie_accounts = std::cmp::max(cfg.accounts / 10, 5) as u64;
    let strie_space = std::cmp::max(cfg.storage_trie_nodes as u64 * 3, 300);
    let mut storage_tries_map: HashMap<B256, Vec<(Nibbles, Option<BranchNodeCompact>)>> =
        HashMap::new();
    for i in 0..cfg.storage_trie_nodes as u64 {
        let acct_idx = (block_number + i) % num_strie_accounts;
        let path_idx = (block_number * 7 + i * 23) % strie_space;
        storage_tries_map
            .entry(addr(acct_idx))
            .or_default()
            .push((nibbles_path(acct_idx, path_idx), Some(sample_node(path_idx))));
    }
    let mut storage_tries: alloy_primitives::map::B256Map<StorageTrieUpdatesSorted> =
        alloy_primitives::map::B256Map::default();
    for (acct, mut nodes) in storage_tries_map {
        nodes.sort_by_key(|(k, _)| k.clone());
        nodes.dedup_by_key(|(k, _)| k.clone());
        storage_tries.insert(
            acct,
            StorageTrieUpdatesSorted {
                is_deleted: false,
                storage_nodes: nodes,
            },
        );
    }

    BlockStateDiff {
        sorted_trie_updates: TrieUpdatesSorted::new(account_nodes, storage_tries),
        sorted_post_state: HashedPostStateSorted::new(accounts, storages),
    }
}

/// Build a `BlockStateDiff` with the default (small-block) config.
fn make_diff(block_number: u64) -> BlockStateDiff {
    make_diff_with_config(block_number, DiffConfig::DEFAULT)
}

fn block_ref(number: u64, hash: B256, parent: B256) -> BlockWithParent {
    BlockWithParent::new(parent, NumHash::new(number, hash))
}

fn block_hash(n: u64) -> B256 {
    addr(10_000 + n)
}

// ---------------------------------------------------------------------------
// V1 helpers
// ---------------------------------------------------------------------------

mod v1 {
    use super::*;

    /// Set up a V1 database with `num_blocks` blocks of data, then return it ready
    /// for pruning.
    pub fn setup(num_blocks: u64) -> (MdbxProofsStorage, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = MdbxProofsStorage::new(dir.path()).unwrap();

        // Initialize with block 0
        {
            let provider = store.initialization_provider().unwrap();
            provider
                .set_initial_state_anchor(alloy_eips::BlockNumHash::new(0, block_hash(0)))
                .unwrap();
            provider.commit_initial_state().unwrap();
            OpProofsInitProvider::commit(provider).unwrap();
        }

        // Write blocks 1..=num_blocks
        let mut parent = block_hash(0);
        for n in 1..=num_blocks {
            let hash = block_hash(n);
            let provider = store.provider_rw().unwrap();
            provider
                .store_trie_updates(block_ref(n, hash, parent), make_diff(n))
                .unwrap();
            OpProofsProviderRw::commit(provider).unwrap();
            parent = hash;
        }

        (store, dir)
    }

    pub fn prune(store: &MdbxProofsStorage, target_block: u64) -> WriteCounts {
        let hash = block_hash(target_block);
        let parent = block_hash(target_block.saturating_sub(1));
        let provider = store.provider_rw().unwrap();
        let counts = provider
            .prune_earliest_state(block_ref(target_block, hash, parent))
            .unwrap();
        OpProofsProviderRw::commit(provider).unwrap();
        counts
    }

    pub fn setup_with_config(num_blocks: u64, cfg: DiffConfig) -> (MdbxProofsStorage, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = MdbxProofsStorage::new(dir.path()).unwrap();
        {
            let provider = store.initialization_provider().unwrap();
            provider
                .set_initial_state_anchor(alloy_eips::BlockNumHash::new(0, block_hash(0)))
                .unwrap();
            provider.commit_initial_state().unwrap();
            OpProofsInitProvider::commit(provider).unwrap();
        }
        let mut parent = block_hash(0);
        for n in 1..=num_blocks {
            let hash = block_hash(n);
            let provider = store.provider_rw().unwrap();
            provider
                .store_trie_updates(block_ref(n, hash, parent), make_diff_with_config(n, cfg))
                .unwrap();
            OpProofsProviderRw::commit(provider).unwrap();
            parent = hash;
        }
        (store, dir)
    }
}

// ---------------------------------------------------------------------------
// V2 helpers
// ---------------------------------------------------------------------------

mod v2 {
    use super::*;

    pub fn setup(num_blocks: u64) -> (MdbxProofsStorageV2, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = MdbxProofsStorageV2::new(dir.path()).unwrap();

        // Initialize with block 0
        {
            let provider = store.initialization_provider().unwrap();
            provider
                .set_initial_state_anchor(alloy_eips::BlockNumHash::new(0, block_hash(0)))
                .unwrap();
            provider.commit_initial_state().unwrap();
            OpProofsInitProvider::commit(provider).unwrap();
        }

        // Write blocks 1..=num_blocks
        let mut parent = block_hash(0);
        for n in 1..=num_blocks {
            let hash = block_hash(n);
            let provider = store.provider_rw().unwrap();
            provider
                .store_trie_updates(block_ref(n, hash, parent), make_diff(n))
                .unwrap();
            OpProofsProviderRw::commit(provider).unwrap();
            parent = hash;
        }

        (store, dir)
    }

    pub fn prune(store: &MdbxProofsStorageV2, target_block: u64) -> WriteCounts {
        let hash = block_hash(target_block);
        let parent = block_hash(target_block.saturating_sub(1));
        let provider = store.provider_rw().unwrap();
        let counts = provider
            .prune_earliest_state(block_ref(target_block, hash, parent))
            .unwrap();
        OpProofsProviderRw::commit(provider).unwrap();
        counts
    }

    pub fn setup_with_config(num_blocks: u64, cfg: DiffConfig) -> (MdbxProofsStorageV2, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = MdbxProofsStorageV2::new(dir.path()).unwrap();
        {
            let provider = store.initialization_provider().unwrap();
            provider
                .set_initial_state_anchor(alloy_eips::BlockNumHash::new(0, block_hash(0)))
                .unwrap();
            provider.commit_initial_state().unwrap();
            OpProofsInitProvider::commit(provider).unwrap();
        }
        let mut parent = block_hash(0);
        for n in 1..=num_blocks {
            let hash = block_hash(n);
            let provider = store.provider_rw().unwrap();
            provider
                .store_trie_updates(block_ref(n, hash, parent), make_diff_with_config(n, cfg))
                .unwrap();
            OpProofsProviderRw::commit(provider).unwrap();
            parent = hash;
        }
        (store, dir)
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_prune(c: &mut Criterion) {
    let mut group = c.benchmark_group("prune_earliest_state");
    group.sample_size(10); // MDBX setup is expensive, fewer samples

    // Prune half the blocks in each scenario
    for num_blocks in [100u64, 500] {
        let prune_target = num_blocks / 2;

        group.bench_with_input(
            BenchmarkId::new("v1", num_blocks),
            &num_blocks,
            |b, &num_blocks| {
                // Setup outside the timed loop — each iteration needs a fresh DB
                b.iter_with_setup(
                    || v1::setup(num_blocks),
                    |(store, _dir)| {
                        black_box(v1::prune(&store, prune_target))
                    },
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("v2", num_blocks),
            &num_blocks,
            |b, &num_blocks| {
                b.iter_with_setup(
                    || v2::setup(num_blocks),
                    |(store, _dir)| {
                        black_box(v2::prune(&store, prune_target))
                    },
                );
            },
        );
    }

    group.finish();
}

fn bench_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_trie_updates");
    group.sample_size(10);

    for num_blocks in [100u64, 500] {
        group.bench_with_input(
            BenchmarkId::new("v1_write", num_blocks),
            &num_blocks,
            |b, &num_blocks| {
                b.iter_with_setup(
                    || {
                        let dir = TempDir::new().unwrap();
                        let store = MdbxProofsStorage::new(dir.path()).unwrap();
                        {
                            let provider = store.initialization_provider().unwrap();
                            provider
                                .set_initial_state_anchor(alloy_eips::BlockNumHash::new(
                                    0,
                                    block_hash(0),
                                ))
                                .unwrap();
                            provider.commit_initial_state().unwrap();
                            OpProofsInitProvider::commit(provider).unwrap();
                        }
                        // Pre-generate diffs
                        let diffs: Vec<_> = (1..=num_blocks).map(|n| make_diff(n)).collect();
                        (store, dir, diffs)
                    },
                    |(store, _dir, diffs)| {
                        let mut parent = block_hash(0);
                        for (i, diff) in diffs.into_iter().enumerate() {
                            let n = (i + 1) as u64;
                            let hash = block_hash(n);
                            let provider = store.provider_rw().unwrap();
                            provider
                                .store_trie_updates(block_ref(n, hash, parent), diff)
                                .unwrap();
                            OpProofsProviderRw::commit(provider).unwrap();
                            parent = hash;
                        }
                    },
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("v2_write", num_blocks),
            &num_blocks,
            |b, &num_blocks| {
                b.iter_with_setup(
                    || {
                        let dir = TempDir::new().unwrap();
                        let store = MdbxProofsStorageV2::new(dir.path()).unwrap();
                        {
                            let provider = store.initialization_provider().unwrap();
                            provider
                                .set_initial_state_anchor(alloy_eips::BlockNumHash::new(
                                    0,
                                    block_hash(0),
                                ))
                                .unwrap();
                            provider.commit_initial_state().unwrap();
                            OpProofsInitProvider::commit(provider).unwrap();
                        }
                        let diffs: Vec<_> = (1..=num_blocks).map(|n| make_diff(n)).collect();
                        (store, dir, diffs)
                    },
                    |(store, _dir, diffs)| {
                        let mut parent = block_hash(0);
                        for (i, diff) in diffs.into_iter().enumerate() {
                            let n = (i + 1) as u64;
                            let hash = block_hash(n);
                            let provider = store.provider_rw().unwrap();
                            provider
                                .store_trie_updates(block_ref(n, hash, parent), diff)
                                .unwrap();
                            OpProofsProviderRw::commit(provider).unwrap();
                            parent = hash;
                        }
                    },
                );
            },
        );
    }

    group.finish();
}

/// Base mainnet benchmark: ~11K changeset entries/block.
///
/// Per-block: 342 hashed accounts, 1660 hashed storages, 1649 account trie
/// nodes, 7340 storage trie nodes (observed Base mainnet averages).
fn bench_base(c: &mut Criterion) {
    // -- prune: 10 blocks written, prune first 5 --
    {
        let mut group = c.benchmark_group("base_prune");
        group.sample_size(10);

        let num_blocks: u64 = 10;
        let prune_target: u64 = 5;

        group.bench_function("v1", |b| {
            b.iter_with_setup(
                || v1::setup_with_config(num_blocks, DiffConfig::BASE),
                |(store, _dir)| black_box(v1::prune(&store, prune_target)),
            );
        });

        group.bench_function("v2", |b| {
            b.iter_with_setup(
                || v2::setup_with_config(num_blocks, DiffConfig::BASE),
                |(store, _dir)| black_box(v2::prune(&store, prune_target)),
            );
        });

        group.finish();
    }

    // -- write: write 10 blocks --
    {
        let mut group = c.benchmark_group("base_write");
        group.sample_size(10);

        let num_blocks: u64 = 10;

        group.bench_function("v1", |b| {
            b.iter_with_setup(
                || {
                    let dir = TempDir::new().unwrap();
                    let store = MdbxProofsStorage::new(dir.path()).unwrap();
                    {
                        let provider = store.initialization_provider().unwrap();
                        provider
                            .set_initial_state_anchor(alloy_eips::BlockNumHash::new(
                                0,
                                block_hash(0),
                            ))
                            .unwrap();
                        provider.commit_initial_state().unwrap();
                        OpProofsInitProvider::commit(provider).unwrap();
                    }
                    let diffs: Vec<_> = (1..=num_blocks)
                        .map(|n| make_diff_with_config(n, DiffConfig::BASE))
                        .collect();
                    (store, dir, diffs)
                },
                |(store, _dir, diffs)| {
                    let mut parent = block_hash(0);
                    for (i, diff) in diffs.into_iter().enumerate() {
                        let n = (i + 1) as u64;
                        let hash = block_hash(n);
                        let provider = store.provider_rw().unwrap();
                        provider
                            .store_trie_updates(block_ref(n, hash, parent), diff)
                            .unwrap();
                        OpProofsProviderRw::commit(provider).unwrap();
                        parent = hash;
                    }
                },
            );
        });

        group.bench_function("v2", |b| {
            b.iter_with_setup(
                || {
                    let dir = TempDir::new().unwrap();
                    let store = MdbxProofsStorageV2::new(dir.path()).unwrap();
                    {
                        let provider = store.initialization_provider().unwrap();
                        provider
                            .set_initial_state_anchor(alloy_eips::BlockNumHash::new(
                                0,
                                block_hash(0),
                            ))
                            .unwrap();
                        provider.commit_initial_state().unwrap();
                        OpProofsInitProvider::commit(provider).unwrap();
                    }
                    let diffs: Vec<_> = (1..=num_blocks)
                        .map(|n| make_diff_with_config(n, DiffConfig::BASE))
                        .collect();
                    (store, dir, diffs)
                },
                |(store, _dir, diffs)| {
                    let mut parent = block_hash(0);
                    for (i, diff) in diffs.into_iter().enumerate() {
                        let n = (i + 1) as u64;
                        let hash = block_hash(n);
                        let provider = store.provider_rw().unwrap();
                        provider
                            .store_trie_updates(block_ref(n, hash, parent), diff)
                            .unwrap();
                        OpProofsProviderRw::commit(provider).unwrap();
                        parent = hash;
                    }
                },
            );
        });

        group.finish();
    }
}

criterion_group!(benches, bench_prune, bench_write, bench_base);
criterion_main!(benches);
