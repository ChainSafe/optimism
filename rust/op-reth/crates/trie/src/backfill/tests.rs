//! Integration tests for [`BackfillJob`].
//!
//! Helpers here mirror `crates/trie/tests/live.rs` because integration-test
//! helpers in `tests/` are not reachable from `src/`. If this duplication grows,
//! consider extracting a shared `test_utils` module behind a feature flag.

use super::{BackfillError, BackfillJob};
use crate::SnapshotInitJob;
use crate::{
    MdbxProofsStorageV2, OpProofsStorageError, OpProofsStore, RethTrieStorageLayout,
    api::{OpProofsProviderRO, OpProofsSnapshotProviderRO},
    db::{
        SnapshotStatus, V2AccountTrieChangeSets, V2AccountsTrie, V2AccountsTrieHistory,
        V2HashedAccountChangeSets, V2HashedAccounts, V2HashedAccountsHistory,
        V2HashedStorageChangeSets, V2HashedStorages, V2HashedStoragesHistory, V2ProofWindow,
        V2StorageTrieChangeSets, V2StoragesTrie, V2StoragesTrieHistory,
    },
    initialize::InitializationJob,
};
use reth_db::{
    DatabaseEnv,
    cursor::DbCursorRO,
    table::Table,
    transaction::DbTx,
};
use alloy_consensus::{BlockHeader, Header, TxEip2930, constants::ETH_TO_WEI};
use alloy_genesis::{Genesis, GenesisAccount};
use alloy_primitives::{Address, B256, Bytes, TxKind, U256, keccak256};
use reth_chainspec::{ChainSpec, ChainSpecBuilder, EthereumHardfork, MAINNET, MIN_TRANSACTION_GAS};
use reth_db::Database;
use reth_db_common::init::init_genesis;
use reth_ethereum_primitives::{Block, BlockBody, Receipt, Transaction, TransactionSigned};
use reth_evm::{ConfigureEvm, execute::Executor};
use reth_evm_ethereum::EthEvmConfig;
use reth_node_api::{NodePrimitives, NodeTypesWithDB};
use reth_primitives_traits::{Block as _, RecoveredBlock};
use reth_provider::{
    BlockWriter as _, DatabaseProviderFactory, ExecutionOutcome, HashedPostStateProvider,
    LatestStateProviderRef, ProviderFactory, StateRootProvider, StorageSettingsCache,
    providers::ProviderNodeTypes, test_utils::create_test_provider_factory_with_chain_spec,
};
use reth_revm::database::StateProviderDatabase;
use secp256k1::{Keypair, Secp256k1, rand::rng};
use serial_test::serial;
use std::sync::Arc;
use tempfile::TempDir;

// ============================ Chain construction helpers ============================

fn create_storage() -> Arc<MdbxProofsStorageV2> {
    let path = TempDir::new().unwrap();
    Arc::new(MdbxProofsStorageV2::new(path.path()).unwrap())
}

fn public_key_to_address(pubkey: secp256k1::PublicKey) -> Address {
    let hash = keccak256(&pubkey.serialize_uncompressed()[1..]);
    Address::from_slice(&hash[12..])
}

fn sign_tx_with_key_pair(key_pair: Keypair, tx: Transaction) -> TransactionSigned {
    use alloy_consensus::SignableTransaction;
    use reth_primitives_traits::crypto::secp256k1::sign_message;
    let secret = B256::from_slice(&key_pair.secret_bytes());
    let sig = sign_message(secret, tx.signature_hash()).unwrap();
    tx.into_signed(sig).into()
}

/// Pre-allocated contract address for storage-write tests.
const STORAGE_CONTRACT: Address = Address::repeat_byte(0xAB);

/// Minimal contract that writes `BLOCKNUMBER` (i.e. current block.number) to
/// storage slot 0:
///
/// ```text
///   0x43        BLOCKNUMBER     push block.number
///   0x60 0x00   PUSH1 0x00      push slot 0
///   0x55        SSTORE          store
///   0x00        STOP
/// ```
const STORAGE_BYTECODE: [u8; 5] = [0x43, 0x60, 0x00, 0x55, 0x00];

fn chain_spec_with_address(address: Address) -> Arc<ChainSpec> {
    Arc::new(
        ChainSpecBuilder::default()
            .chain(MAINNET.chain)
            .genesis(Genesis {
                alloc: [
                    (
                        address,
                        GenesisAccount {
                            balance: U256::from(10 * ETH_TO_WEI),
                            ..Default::default()
                        },
                    ),
                    (
                        STORAGE_CONTRACT,
                        GenesisAccount {
                            code: Some(Bytes::from_static(&STORAGE_BYTECODE)),
                            ..Default::default()
                        },
                    ),
                ]
                .into(),
                ..MAINNET.genesis.clone()
            })
            .paris_activated()
            .build(),
    )
}

/// Construct an unsealed block with a single simple transfer.
fn build_transfer_block(
    block_number: u64,
    parent_hash: B256,
    chain_spec: &Arc<ChainSpec>,
    key_pair: Keypair,
    nonce: u64,
    recipient: Address,
) -> RecoveredBlock<Block> {
    let tx = sign_tx_with_key_pair(
        key_pair,
        TxEip2930 {
            chain_id: chain_spec.chain.id(),
            nonce,
            gas_limit: MIN_TRANSACTION_GAS,
            gas_price: 1_500_000_000,
            to: TxKind::Call(recipient),
            value: U256::from(1),
            ..Default::default()
        }
        .into(),
    );
    Block {
        header: Header {
            parent_hash,
            receipts_root: alloy_primitives::b256!(
                "0xd3a6acf9a244d78b33831df95d472c4128ea85bf079a1d41e32ed0b7d2244c9e"
            ),
            difficulty: chain_spec.fork(EthereumHardfork::Paris).ttd().expect("Paris TTD"),
            number: block_number,
            gas_limit: MIN_TRANSACTION_GAS,
            gas_used: MIN_TRANSACTION_GAS,
            state_root: B256::ZERO, // filled in by execute_block
            ..Default::default()
        },
        body: BlockBody { transactions: vec![tx], ..Default::default() },
    }
    .try_into_recovered()
    .unwrap()
}

fn execute_block<N>(
    block: &mut RecoveredBlock<Block>,
    provider_factory: &ProviderFactory<N>,
    chain_spec: &Arc<ChainSpec>,
) -> reth_evm::execute::BlockExecutionOutput<Receipt>
where
    N: ProviderNodeTypes<
            Primitives: NodePrimitives<Block = Block, BlockBody = BlockBody, Receipt = Receipt>,
        > + NodeTypesWithDB,
{
    let provider = provider_factory.provider().unwrap();
    let db = StateProviderDatabase::new(LatestStateProviderRef::new(&provider));
    let evm_config = EthEvmConfig::ethereum(chain_spec.clone());
    let block_executor = evm_config.batch_executor(db);
    let execution_result = block_executor.execute(block).unwrap();

    let hashed_state =
        LatestStateProviderRef::new(&provider).hashed_post_state(&execution_result.state);
    let state_root = LatestStateProviderRef::new(&provider).state_root(hashed_state).unwrap();
    block.set_state_root(state_root);
    execution_result
}

fn commit_block_to_database<N>(
    block: &RecoveredBlock<Block>,
    execution_output: &reth_evm::execute::BlockExecutionOutput<Receipt>,
    provider_factory: &ProviderFactory<N>,
) where
    N: ProviderNodeTypes<
            Primitives: NodePrimitives<Block = Block, BlockBody = BlockBody, Receipt = Receipt>,
        > + NodeTypesWithDB,
{
    let execution_outcome = ExecutionOutcome {
        bundle: execution_output.state.clone(),
        receipts: vec![execution_output.receipts.clone()],
        first_block: block.number(),
        requests: vec![execution_output.requests.clone()],
    };
    let state_provider = provider_factory.provider().unwrap();
    let hashed_state = HashedPostStateProvider::hashed_post_state(
        &LatestStateProviderRef::new(&state_provider),
        &execution_output.state,
    );
    let provider_rw = provider_factory.provider_rw().unwrap();
    provider_rw
        .append_blocks_with_state(
            vec![block.clone()],
            &execution_outcome,
            hashed_state.into_sorted(),
        )
        .unwrap();
    provider_rw.commit().unwrap();
}

/// Construct an unsealed block whose sole tx calls [`STORAGE_CONTRACT`],
/// triggering an SSTORE of `block.number` into slot 0 of the contract's storage.
///
/// Gas accounting: the executor recomputes `gas_used` against the actual EVM
/// trace, so we deliberately set `gas_limit == gas_used` to a value large
/// enough to cover both the 21 000-gas tx base cost and the worst-case cold
/// SSTORE (~22 100 gas).
fn build_storage_call_block(
    block_number: u64,
    parent_hash: B256,
    chain_spec: &Arc<ChainSpec>,
    key_pair: Keypair,
    nonce: u64,
) -> RecoveredBlock<Block> {
    const CALL_GAS_LIMIT: u64 = 100_000;
    let tx = sign_tx_with_key_pair(
        key_pair,
        TxEip2930 {
            chain_id: chain_spec.chain.id(),
            nonce,
            gas_limit: CALL_GAS_LIMIT,
            gas_price: 1_500_000_000,
            to: TxKind::Call(STORAGE_CONTRACT),
            value: U256::ZERO,
            ..Default::default()
        }
        .into(),
    );
    Block {
        header: Header {
            parent_hash,
            receipts_root: alloy_primitives::b256!(
                "0xd3a6acf9a244d78b33831df95d472c4128ea85bf079a1d41e32ed0b7d2244c9e"
            ),
            difficulty: chain_spec.fork(EthereumHardfork::Paris).ttd().expect("Paris TTD"),
            number: block_number,
            gas_limit: CALL_GAS_LIMIT,
            gas_used: CALL_GAS_LIMIT,
            state_root: B256::ZERO,
            ..Default::default()
        },
        body: BlockBody { transactions: vec![tx], ..Default::default() },
    }
    .try_into_recovered()
    .unwrap()
}

/// Build a chain of `num_blocks` simple transfer blocks on top of a freshly
/// initialized genesis, then initialize the v2 proofs storage at the latest
/// block. Returns the provider factory, the storage, and the latest
/// (number, hash) pair.
fn build_chain_and_initialize_storage(
    num_blocks: u64,
) -> (
    ProviderFactory<reth_provider::test_utils::MockNodeTypesWithDB>,
    Arc<MdbxProofsStorageV2>,
    u64,
    B256,
) {
    let secp = Secp256k1::new();
    let key_pair = Keypair::new(&secp, &mut rng());
    let sender = public_key_to_address(key_pair.public_key());

    let chain_spec = chain_spec_with_address(sender);
    let provider_factory = create_test_provider_factory_with_chain_spec(chain_spec.clone());
    init_genesis(&provider_factory).unwrap();

    let recipient = Address::repeat_byte(0x42);
    let mut last_hash = chain_spec.genesis_hash();
    let mut last_number = 0u64;
    for n in 1..=num_blocks {
        let mut block = build_transfer_block(n, last_hash, &chain_spec, key_pair, n - 1, recipient);
        let exec = execute_block(&mut block, &provider_factory, &chain_spec);
        commit_block_to_database(&block, &exec, &provider_factory);
        last_hash = block.hash();
        last_number = n;
    }

    let storage = create_storage();
    {
        let trie_layout = if provider_factory.cached_storage_settings().is_v2() {
            RethTrieStorageLayout::Packed
        } else {
            RethTrieStorageLayout::Legacy
        };
        let tx = provider_factory.db_ref().tx().unwrap();
        InitializationJob::new(storage.clone(), tx, trie_layout)
            .run(last_number, last_hash)
            .unwrap();
    }

    (provider_factory, storage, last_number, last_hash)
}

/// Like [`build_chain_and_initialize_storage`] but every block calls
/// [`STORAGE_CONTRACT`], so each block produces hashed-storage changesets in
/// addition to the account-level ones.
fn build_chain_with_storage_writes_and_initialize_storage(
    num_blocks: u64,
) -> (
    ProviderFactory<reth_provider::test_utils::MockNodeTypesWithDB>,
    Arc<MdbxProofsStorageV2>,
    u64,
    B256,
) {
    let secp = Secp256k1::new();
    let key_pair = Keypair::new(&secp, &mut rng());
    let sender = public_key_to_address(key_pair.public_key());

    let chain_spec = chain_spec_with_address(sender);
    let provider_factory = create_test_provider_factory_with_chain_spec(chain_spec.clone());
    init_genesis(&provider_factory).unwrap();

    let mut last_hash = chain_spec.genesis_hash();
    let mut last_number = 0u64;
    for n in 1..=num_blocks {
        let mut block = build_storage_call_block(n, last_hash, &chain_spec, key_pair, n - 1);
        let exec = execute_block(&mut block, &provider_factory, &chain_spec);
        commit_block_to_database(&block, &exec, &provider_factory);
        last_hash = block.hash();
        last_number = n;
    }

    let storage = create_storage();
    {
        let trie_layout = if provider_factory.cached_storage_settings().is_v2() {
            RethTrieStorageLayout::Packed
        } else {
            RethTrieStorageLayout::Legacy
        };
        let tx = provider_factory.db_ref().tx().unwrap();
        InitializationJob::new(storage.clone(), tx, trie_layout)
            .run(last_number, last_hash)
            .unwrap();
    }

    (provider_factory, storage, last_number, last_hash)
}

// ============================ Tests ============================

#[test]
#[serial]
fn run_is_noop_when_target_at_or_above_earliest() {
    // Build a chain of 3 blocks; storage initialized at block 3 (earliest = 3).
    let (provider_factory, storage, latest_num, latest_hash) =
        build_chain_and_initialize_storage(3);

    // target == earliest: no-op.
    {
        let provider = provider_factory.database_provider_ro().unwrap();
        BackfillJob::new(provider, storage.clone()).run(latest_num).unwrap();
        let ro = storage.provider_ro().unwrap();
        assert_eq!(ro.get_earliest_block_number().unwrap(), Some((latest_num, latest_hash)));
    }

    // target > earliest: also no-op.
    {
        let provider = provider_factory.database_provider_ro().unwrap();
        BackfillJob::new(provider, storage.clone()).run(latest_num + 100).unwrap();
        let ro = storage.provider_ro().unwrap();
        assert_eq!(ro.get_earliest_block_number().unwrap(), Some((latest_num, latest_hash)));
    }
}

#[test]
#[serial]
fn run_errors_when_storage_uninitialized() {
    let secp = Secp256k1::new();
    let key_pair = Keypair::new(&secp, &mut rng());
    let chain_spec = chain_spec_with_address(public_key_to_address(key_pair.public_key()));
    let provider_factory = create_test_provider_factory_with_chain_spec(chain_spec);
    init_genesis(&provider_factory).unwrap();

    // Storage created but never initialized — no earliest marker.
    let storage = create_storage();
    let provider = provider_factory.database_provider_ro().unwrap();
    let err = BackfillJob::new(provider, storage).run(0).unwrap_err();
    assert!(
        matches!(err, BackfillError::Storage(OpProofsStorageError::NoBlocksFound)),
        "expected NoBlocksFound, got {err:?}"
    );
}

#[test]
#[serial]
fn run_extends_window_backward_multi_block() {
    // 5-block chain — exercises descending iteration across multiple
    // `BackfillContext::step` calls.
    let (provider_factory, storage, latest_num, latest_hash) =
        build_chain_and_initialize_storage(5);

    {
        let ro = storage.provider_ro().unwrap();
        assert_eq!(ro.get_earliest_block_number().unwrap(), Some((latest_num, latest_hash)));
    }

    {
        let provider = provider_factory.database_provider_ro().unwrap();
        BackfillJob::new(provider, storage.clone()).run(0).unwrap();
    }

    let provider = provider_factory.database_provider_ro().unwrap();
    let genesis_hash = reth_provider::BlockHashReader::block_hash(&provider, 0).unwrap().unwrap();
    let ro = storage.provider_ro().unwrap();
    assert_eq!(ro.get_earliest_block_number().unwrap(), Some((0, genesis_hash)));
}

#[test]
#[serial]
fn run_extends_window_backward() {
    // Smallest possible case: 1-block chain, single backfill step from 1 → 0.
    let (provider_factory, storage, latest_num, latest_hash) =
        build_chain_and_initialize_storage(1);

    // Sanity: earliest starts at the latest block.
    {
        let ro = storage.provider_ro().unwrap();
        assert_eq!(ro.get_earliest_block_number().unwrap(), Some((latest_num, latest_hash)));
    }

    // Backfill all the way down to block 0 (genesis).
    {
        let provider = provider_factory.database_provider_ro().unwrap();
        BackfillJob::new(provider, storage.clone()).run(0).unwrap();
    }

    // Earliest should now point at block 0 (the genesis hash).
    let provider = provider_factory.database_provider_ro().unwrap();
    let genesis_hash = reth_provider::BlockHashReader::block_hash(&provider, 0).unwrap().unwrap();
    let ro = storage.provider_ro().unwrap();
    assert_eq!(ro.get_earliest_block_number().unwrap(), Some((0, genesis_hash)));
}

#[test]
#[serial]
fn snapshot_init_validates_against_header_and_marks_ready() {
    let (provider_factory, storage, latest_num, latest_hash) =
        build_chain_with_storage_writes_and_initialize_storage(3);

    // Before init: meta is absent.
    {
        let ro = storage.provider_ro().unwrap();
        assert_eq!(ro.snapshot_meta().unwrap(), None);
    }

    let outcome = {
        let provider = provider_factory.database_provider_ro().unwrap();
        SnapshotInitJob::new(provider, storage.clone()).run(latest_num).unwrap()
    };

    // Meta points at latest, status = Ready. The fact that we reached `Ready`
    // proves the state-root validation passed against the header's `state_root`.
    // For tiny chains both copy counts can be zero (few accounts → leaves only,
    // no branch nodes promoted into the persistent trie table) — the validation
    // is the real assertion, not the count.
    assert_eq!(outcome.meta.earliest.number, latest_num);
    assert_eq!(outcome.meta.earliest.hash, latest_hash);
    assert_eq!(outcome.meta.status, SnapshotStatus::Ready);

    // Reading back through the provider sees the same meta.
    {
        let ro = storage.provider_ro().unwrap();
        assert_eq!(ro.snapshot_meta().unwrap(), Some(outcome.meta));
    }

    // Re-running refuses because a snapshot already exists.
    let err = {
        let provider = provider_factory.database_provider_ro().unwrap();
        SnapshotInitJob::new(provider, storage).run(latest_num).unwrap_err()
    };
    assert!(
        matches!(err, BackfillError::SnapshotAlreadyExists { .. }),
        "expected SnapshotAlreadyExists, got {err:?}"
    );
}

#[test]
#[serial]
fn snapshot_init_resumes_partial_build() {
    // Simulate an interrupted init: build a chain, run SnapshotInitJob once,
    // then forcibly downgrade meta to `Building` (mimicking a crash before the
    // final Ready commit). Re-run and verify it finishes cleanly with Ready,
    // anchor unchanged, and the snapshot tables exactly mirror the source.
    use crate::api::OpProofsSnapshotProviderRW;
    use crate::db::{SnapshotMeta, V2AccountsTrie, V2AccountsTrieSnapshot, V2StoragesTrie, V2StoragesTrieSnapshot};
    use alloy_eips::BlockNumHash;
    use reth_db::Database;

    let (provider_factory, storage, latest_num, latest_hash) =
        build_chain_with_storage_writes_and_initialize_storage(3);

    // First run — completes normally to Ready.
    {
        let provider = provider_factory.database_provider_ro().unwrap();
        SnapshotInitJob::new(provider, storage.clone()).run(latest_num).unwrap();
    }
    // Sanity: ready + anchor matches latest.
    {
        let meta = storage.provider_ro().unwrap().snapshot_meta().unwrap().unwrap();
        assert_eq!(meta.status, SnapshotStatus::Ready);
        assert_eq!(meta.earliest.number, latest_num);
    }

    // Force the meta back to Building (simulates crash post-data, pre-final-commit).
    // Write directly via the DB tx — the trait surface doesn't (and shouldn't)
    // expose a way to downgrade `Ready → Building` while preserving data.
    {
        use crate::db::{SnapshotMetaKey, V2TrieSnapshotMeta};
        use reth_db::{cursor::DbCursorRW, transaction::DbTxMut};
        let tx = storage.env().tx_mut().unwrap();
        let mut cur = tx.cursor_write::<V2TrieSnapshotMeta>().unwrap();
        cur.upsert(
            SnapshotMetaKey::Singleton,
            &SnapshotMeta::new(
                BlockNumHash::new(latest_num, latest_hash),
                SnapshotStatus::Building,
            ),
        )
        .unwrap();
        tx.commit().unwrap();
    }

    // Resume: SnapshotInitJob detects Building+matching-anchor and resumes.
    // Because the chunks were all already written, the resume should walk the
    // source past the destination's last-key (finding nothing), validate, and
    // flip to Ready.
    let outcome = {
        let provider = provider_factory.database_provider_ro().unwrap();
        SnapshotInitJob::new(provider, storage.clone()).run(latest_num).unwrap()
    };
    // No new rows were copied — everything was already there.
    assert_eq!(outcome.account_nodes_copied, 0);
    assert_eq!(outcome.storage_nodes_copied, 0);
    assert_eq!(outcome.meta.status, SnapshotStatus::Ready);
    assert_eq!(outcome.meta.earliest.number, latest_num);

    // Source vs destination row count must match (snapshot mirrors current state).
    let env = storage.env();
    let tx = env.tx().unwrap();
    let acc_src: Vec<_> = tx
        .cursor_read::<V2AccountsTrie>()
        .unwrap()
        .walk(None)
        .unwrap()
        .map(Result::unwrap)
        .collect();
    let acc_dst: Vec<_> = tx
        .cursor_read::<V2AccountsTrieSnapshot>()
        .unwrap()
        .walk(None)
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(acc_src, acc_dst, "snapshot account trie must mirror source");

    let stor_src: Vec<_> = tx
        .cursor_read::<V2StoragesTrie>()
        .unwrap()
        .walk(None)
        .unwrap()
        .map(Result::unwrap)
        .collect();
    let stor_dst: Vec<_> = tx
        .cursor_read::<V2StoragesTrieSnapshot>()
        .unwrap()
        .walk(None)
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(stor_src, stor_dst, "snapshot storage trie must mirror source");
}

#[test]
#[serial]
fn snapshot_init_refuses_resume_when_anchor_moved() {
    // Stand up a partial Building snapshot at one block, then move the
    // proofs-window `earliest` backward via a merge-walk backfill step.
    // Re-running init must refuse with `SnapshotResumeDriftDetected` because
    // the anchor no longer matches `current earliest`.
    use crate::api::OpProofsSnapshotInitProvider;
    use alloy_eips::BlockNumHash;

    let (provider_factory, storage, latest_num, latest_hash) =
        build_chain_and_initialize_storage(3);

    // Plant a Building meta with the current `earliest` (= `latest` for a
    // fresh proofs storage) via the init-provider's `set_snapshot_init_anchor`.
    {
        use crate::OpProofsSnapshotStore;
        let sp = storage.snapshot_provider().unwrap();
        sp.set_snapshot_init_anchor(BlockNumHash::new(latest_num, latest_hash)).unwrap();
        OpProofsSnapshotInitProvider::commit(sp).unwrap();
    }

    // Move `earliest` backward by running one merge-walk backfill step.
    {
        let provider = provider_factory.database_provider_ro().unwrap();
        BackfillJob::new(provider, storage.clone()).run(latest_num - 1).unwrap();
    }

    // Re-run init targeting the new `earliest` (latest_num - 1). The existing
    // Building meta is at latest_num, so the targets don't match → drift.
    let provider = provider_factory.database_provider_ro().unwrap();
    let err = SnapshotInitJob::new(provider, storage).run(latest_num - 1).unwrap_err();
    assert!(
        matches!(err, BackfillError::SnapshotResumeDriftDetected { .. }),
        "expected SnapshotResumeDriftDetected, got {err:?}"
    );
}

#[test]
#[serial]
fn snapshot_init_errors_when_storage_uninitialized() {
    let secp = Secp256k1::new();
    let key_pair = Keypair::new(&secp, &mut rng());
    let chain_spec = chain_spec_with_address(public_key_to_address(key_pair.public_key()));
    let provider_factory = create_test_provider_factory_with_chain_spec(chain_spec);
    init_genesis(&provider_factory).unwrap();

    let storage = create_storage();
    let provider = provider_factory.database_provider_ro().unwrap();
    let err = SnapshotInitJob::new(provider, storage).run(0).unwrap_err();
    assert!(
        matches!(err, BackfillError::SnapshotInitNoEarliest),
        "expected SnapshotInitNoEarliest, got {err:?}"
    );
}

#[test]
#[serial]
fn run_with_snapshot_extends_window_backward() {
    // 5-block chain with storage writes — exercises both account and storage
    // trie reverts on the snapshot path.
    let (provider_factory, storage, latest_num, latest_hash) =
        build_chain_with_storage_writes_and_initialize_storage(5);

    // Build the snapshot at `latest`.
    {
        let provider = provider_factory.database_provider_ro().unwrap();
        SnapshotInitJob::new(provider, storage.clone()).run(latest_num).unwrap();
    }

    // Sanity: snapshot Ready at latest.
    {
        let ro = storage.provider_ro().unwrap();
        let meta = ro.snapshot_meta().unwrap().unwrap();
        assert_eq!(meta.status, SnapshotStatus::Ready);
        assert_eq!(meta.earliest.number, latest_num);
    }

    // Run snapshot backfill all the way down to genesis.
    {
        let provider = provider_factory.database_provider_ro().unwrap();
        BackfillJob::new(provider, storage.clone()).run_with_snapshot(0).unwrap();
    }

    // Proofs window earliest now at 0 and snapshot tracks block 0 too.
    let provider = provider_factory.database_provider_ro().unwrap();
    let genesis_hash = reth_provider::BlockHashReader::block_hash(&provider, 0).unwrap().unwrap();
    let ro = storage.provider_ro().unwrap();
    assert_eq!(ro.get_earliest_block_number().unwrap(), Some((0, genesis_hash)));
    let meta = ro.snapshot_meta().unwrap().unwrap();
    assert_eq!(meta.status, SnapshotStatus::Ready);
    assert_eq!(meta.earliest.number, 0);
    assert_eq!(meta.earliest.hash, genesis_hash);

    // Reject reuse on the no-op edge: target == latest after backfill is a
    // no-op (covered by `run_is_noop_when_target_at_or_above_earliest` for the
    // merge-walk path). Confirm it for the snapshot path too.
    let _ = latest_hash;
    {
        let provider = provider_factory.database_provider_ro().unwrap();
        BackfillJob::new(provider, storage.clone()).run_with_snapshot(0).unwrap();
    }
}

#[test]
#[serial]
fn run_auto_builds_snapshot_when_missing_and_proceeds() {
    // Fresh init: earliest == latest, no snapshot. `run_auto` should build one
    // and then drive the backfill down to target.
    let (provider_factory, storage, _, _) =
        build_chain_with_storage_writes_and_initialize_storage(4);

    // Sanity: no snapshot present yet.
    {
        let ro = storage.provider_ro().unwrap();
        assert_eq!(ro.snapshot_meta().unwrap(), None);
    }

    {
        let provider = provider_factory.database_provider_ro().unwrap();
        BackfillJob::new(provider, storage.clone()).run_auto(0).unwrap();
    }

    // Both proofs-window earliest and snapshot meta now anchor block 0.
    let provider = provider_factory.database_provider_ro().unwrap();
    let genesis_hash = reth_provider::BlockHashReader::block_hash(&provider, 0).unwrap().unwrap();
    let ro = storage.provider_ro().unwrap();
    assert_eq!(ro.get_earliest_block_number().unwrap(), Some((0, genesis_hash)));
    let meta = ro.snapshot_meta().unwrap().unwrap();
    assert_eq!(meta.status, SnapshotStatus::Ready);
    assert_eq!(meta.earliest.number, 0);
}

#[test]
#[serial]
fn run_auto_builds_snapshot_at_earliest_below_latest() {
    // 3-block chain, init storage, then run merge-walk backfill so
    // `earliest` drops below `latest`. `run_auto` should then auto-build a
    // snapshot anchored at the current `earliest` (using the merge-walk
    // cursors) and proceed to backfill the remaining blocks.
    let (provider_factory, storage, latest_num, _) = build_chain_and_initialize_storage(3);

    {
        let provider = provider_factory.database_provider_ro().unwrap();
        // Backfill one step: earliest = latest_num - 1.
        BackfillJob::new(provider, storage.clone()).run(latest_num - 1).unwrap();
    }

    {
        let provider = provider_factory.database_provider_ro().unwrap();
        BackfillJob::new(provider, storage.clone()).run_auto(0).unwrap();
    }

    let provider = provider_factory.database_provider_ro().unwrap();
    let genesis_hash = reth_provider::BlockHashReader::block_hash(&provider, 0).unwrap().unwrap();
    let ro = storage.provider_ro().unwrap();
    assert_eq!(ro.get_earliest_block_number().unwrap(), Some((0, genesis_hash)));
    let meta = ro.snapshot_meta().unwrap().unwrap();
    assert_eq!(meta.status, SnapshotStatus::Ready);
    assert_eq!(meta.earliest.number, 0);
}

/// Walk both DBs' rows of `T` in parallel and assert each `(key, value)` pair
/// matches. Reports the row index of any divergence for diagnostics.
fn assert_tables_equal<T>(a: &DatabaseEnv, b: &DatabaseEnv, table_name: &'static str)
where
    T: Table,
    T::Key: PartialEq + std::fmt::Debug,
    T::Value: PartialEq + std::fmt::Debug,
{
    let tx_a = a.tx().unwrap();
    let tx_b = b.tx().unwrap();
    let mut ca = tx_a.cursor_read::<T>().unwrap();
    let mut cb = tx_b.cursor_read::<T>().unwrap();
    let mut row = 0usize;
    let mut ea = ca.first().unwrap();
    let mut eb = cb.first().unwrap();
    loop {
        match (ea.as_ref(), eb.as_ref()) {
            (Some((ka, va)), Some((kb, vb))) => {
                assert_eq!(ka, kb, "{table_name}: key mismatch at row {row}");
                assert_eq!(va, vb, "{table_name}: value mismatch at row {row} (key={ka:?})");
            }
            (None, None) => break,
            (Some((ka, _)), None) => {
                panic!("{table_name}: A has extra rows starting at row {row} (key={ka:?})")
            }
            (None, Some((kb, _))) => {
                panic!("{table_name}: B has extra rows starting at row {row} (key={kb:?})")
            }
        }
        ea = ca.next().unwrap();
        eb = cb.next().unwrap();
        row += 1;
    }
}

#[test]
#[serial]
fn run_auto_and_merge_walk_produce_identical_storage() {
    // Build one chain, then initialize two storages against it. Run merge-walk
    // backfill on A and snapshot backfill on B. Every persisted row in the
    // tables `prepend_block` touches — changesets, history bitmaps, proof
    // window markers — must match byte-for-byte. The snapshot tables
    // themselves are intentionally not compared (A doesn't have them).
    let secp = Secp256k1::new();
    let key_pair = Keypair::new(&secp, &mut rng());
    let sender = public_key_to_address(key_pair.public_key());
    let chain_spec = chain_spec_with_address(sender);
    let provider_factory = create_test_provider_factory_with_chain_spec(chain_spec.clone());
    init_genesis(&provider_factory).unwrap();

    let num_blocks = 5;
    let mut last_hash = chain_spec.genesis_hash();
    let mut last_number = 0u64;
    for n in 1..=num_blocks {
        let mut block = build_storage_call_block(n, last_hash, &chain_spec, key_pair, n - 1);
        let exec = execute_block(&mut block, &provider_factory, &chain_spec);
        commit_block_to_database(&block, &exec, &provider_factory);
        last_hash = block.hash();
        last_number = n;
    }

    // Two independent storages, both initialized at the same `latest`.
    let storage_a = create_storage();
    let storage_b = create_storage();
    let trie_layout = if provider_factory.cached_storage_settings().is_v2() {
        RethTrieStorageLayout::Packed
    } else {
        RethTrieStorageLayout::Legacy
    };
    {
        let tx = provider_factory.db_ref().tx().unwrap();
        InitializationJob::new(storage_a.clone(), tx, trie_layout)
            .run(last_number, last_hash)
            .unwrap();
    }
    {
        let tx = provider_factory.db_ref().tx().unwrap();
        InitializationJob::new(storage_b.clone(), tx, trie_layout)
            .run(last_number, last_hash)
            .unwrap();
    }

    // Sanity: post-init both storages have identical contents.
    assert_tables_equal::<V2AccountsTrie>(storage_a.env(), storage_b.env(), "V2AccountsTrie");
    assert_tables_equal::<V2StoragesTrie>(storage_a.env(), storage_b.env(), "V2StoragesTrie");
    assert_tables_equal::<V2HashedAccounts>(storage_a.env(), storage_b.env(), "V2HashedAccounts");
    assert_tables_equal::<V2HashedStorages>(storage_a.env(), storage_b.env(), "V2HashedStorages");

    // A: merge-walk backfill.
    {
        let provider = provider_factory.database_provider_ro().unwrap();
        BackfillJob::new(provider, storage_a.clone()).run(0).unwrap();
    }
    // B: snapshot backfill (auto-builds the snapshot internally).
    {
        let provider = provider_factory.database_provider_ro().unwrap();
        BackfillJob::new(provider, storage_b.clone()).run_auto(0).unwrap();
    }

    // Both reached the same earliest.
    {
        let ro_a = storage_a.provider_ro().unwrap();
        let ro_b = storage_b.provider_ro().unwrap();
        assert_eq!(ro_a.get_earliest_block_number().unwrap(), ro_b.get_earliest_block_number().unwrap());
    }

    // Compare every table prepend_block touches.
    let (a, b) = (storage_a.env(), storage_b.env());
    assert_tables_equal::<V2AccountTrieChangeSets>(a, b, "V2AccountTrieChangeSets");
    assert_tables_equal::<V2StorageTrieChangeSets>(a, b, "V2StorageTrieChangeSets");
    assert_tables_equal::<V2HashedAccountChangeSets>(a, b, "V2HashedAccountChangeSets");
    assert_tables_equal::<V2HashedStorageChangeSets>(a, b, "V2HashedStorageChangeSets");
    assert_tables_equal::<V2AccountsTrieHistory>(a, b, "V2AccountsTrieHistory");
    assert_tables_equal::<V2StoragesTrieHistory>(a, b, "V2StoragesTrieHistory");
    assert_tables_equal::<V2HashedAccountsHistory>(a, b, "V2HashedAccountsHistory");
    assert_tables_equal::<V2HashedStoragesHistory>(a, b, "V2HashedStoragesHistory");
    assert_tables_equal::<V2ProofWindow>(a, b, "V2ProofWindow");
    // Current-state tables must remain unchanged from init.
    assert_tables_equal::<V2AccountsTrie>(a, b, "V2AccountsTrie");
    assert_tables_equal::<V2StoragesTrie>(a, b, "V2StoragesTrie");
    assert_tables_equal::<V2HashedAccounts>(a, b, "V2HashedAccounts");
    assert_tables_equal::<V2HashedStorages>(a, b, "V2HashedStorages");
}

#[test]
#[serial]
fn run_with_snapshot_errors_when_snapshot_missing() {
    // Build a chain + initialize storage but do NOT build a snapshot.
    let (provider_factory, storage, _, _) = build_chain_and_initialize_storage(2);

    let provider = provider_factory.database_provider_ro().unwrap();
    let err = BackfillJob::new(provider, storage).run_with_snapshot(0).unwrap_err();
    assert!(
        matches!(err, BackfillError::SnapshotNotAligned { actual_status: None, .. }),
        "expected SnapshotNotAligned with no existing snapshot, got {err:?}"
    );
}

#[test]
#[serial]
fn run_extends_window_backward_with_storage_writes() {
    // Every block calls `STORAGE_CONTRACT`, writing `block.number` to slot 0.
    // This exercises the backfill code paths that are silent in plain-transfer
    // tests:
    //   - `V2HashedStorageChangeSets` / `V2HashedStoragesHistory` writes during `prepend_block`
    //     (the slot value changes every block).
    //   - Storage-side reconstruction via `V2StorageCursor` at each historical block during the
    //     in-job `StateRoot::overlay_root` validation.
    let (provider_factory, storage, latest_num, latest_hash) =
        build_chain_with_storage_writes_and_initialize_storage(5);

    {
        let ro = storage.provider_ro().unwrap();
        assert_eq!(ro.get_earliest_block_number().unwrap(), Some((latest_num, latest_hash)));
    }

    {
        let provider = provider_factory.database_provider_ro().unwrap();
        BackfillJob::new(provider, storage.clone()).run(0).unwrap();
    }

    let provider = provider_factory.database_provider_ro().unwrap();
    let genesis_hash = reth_provider::BlockHashReader::block_hash(&provider, 0).unwrap().unwrap();
    let ro = storage.provider_ro().unwrap();
    assert_eq!(ro.get_earliest_block_number().unwrap(), Some((0, genesis_hash)));
}
