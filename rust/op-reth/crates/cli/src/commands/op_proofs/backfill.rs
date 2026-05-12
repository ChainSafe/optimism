//! Command that backfills OP proofs storage to an older earliest block.

use clap::Parser;
use reth_cli::chainspec::ChainSpecParser;
use reth_cli_commands::common::{AccessRights, CliNodeTypes, Environment, EnvironmentArgs};
use reth_node_core::version::version_metadata;
use reth_optimism_chainspec::OpChainSpec;
use reth_optimism_node::args::ProofsStorageVersion;
use reth_optimism_primitives::OpPrimitives;
use reth_optimism_trie::{
    BackfillJob, OpProofsProviderRO, OpProofsSnapshotProviderRO, OpProofsSnapshotProviderRW,
    OpProofsSnapshotStore, OpProofsStore,
    db::MdbxProofsStorageV2,
};
use reth_provider::{
    BlockHashReader, BlockNumReader, ChangeSetReader, DBProvider, DatabaseProviderFactory,
    HeaderProvider, StageCheckpointReader, StorageChangeSetReader, StorageSettingsCache,
};
use std::{path::PathBuf, sync::Arc};
use tracing::info;

/// Backfills the proofs storage to an older earliest block.
#[derive(Debug, Parser)]
pub struct BackfillCommand<C: ChainSpecParser> {
    #[command(flatten)]
    env: EnvironmentArgs<C>,

    /// The path to the storage DB for proofs history.
    #[arg(
        long = "proofs-history.storage-path",
        value_name = "PROOFS_HISTORY_STORAGE_PATH",
        required = true
    )]
    pub storage_path: PathBuf,

    /// Target earliest block number after backfill.
    #[arg(long = "proofs-history.target-earliest-block", value_name = "TARGET_EARLIEST_BLOCK")]
    pub target_earliest_block: u64,

    /// Storage schema version. Must match the version used when starting the node.
    #[arg(
        long = "proofs-history.storage-version",
        value_name = "PROOFS_HISTORY_STORAGE_VERSION",
        default_value = "v1"
    )]
    pub storage_version: ProofsStorageVersion,

    /// Use the trie-snapshot fast path.
    ///
    /// When set, the backfill auto-manages a [`SnapshotStatus::Ready`] snapshot
    /// and uses it for the per-block compute phase, eliminating the V2 merge-walk's
    /// per-key `find_source` work. If no snapshot exists and the proofs window
    /// is anchored at `latest`, one is built on the fly.
    ///
    /// [`SnapshotStatus::Ready`]: reth_optimism_trie::db::SnapshotStatus::Ready
    #[arg(long = "proofs-history.snapshot", default_value_t = false)]
    pub snapshot: bool,
}

impl<C: ChainSpecParser<ChainSpec = OpChainSpec>> BackfillCommand<C> {
    /// Execute [`BackfillCommand`].
    pub async fn execute<N: CliNodeTypes<ChainSpec = C::ChainSpec, Primitives = OpPrimitives>>(
        self,
        runtime: reth_tasks::Runtime,
    ) -> eyre::Result<()> {
        info!(target: "reth::cli", "reth {} starting", version_metadata().short_version);
        info!(target: "reth::cli", "Backfilling OP proofs storage at: {:?}", self.storage_path);

        let Environment { provider_factory, .. } = self.env.init::<N>(AccessRights::RO, runtime)?;

        match self.storage_version {
            ProofsStorageVersion::V1 => {
                return Err(eyre::eyre!(
                    "Backfill is not supported for V1 proofs storage. \
                     Re-run with --proofs-history.storage-version v2."
                ));
            }
            ProofsStorageVersion::V2 => {
                let storage: Arc<MdbxProofsStorageV2> = Arc::new(
                    MdbxProofsStorageV2::new(&self.storage_path)
                        .map_err(|e| eyre::eyre!("Failed to create MdbxProofsStorageV2: {e}"))?,
                );
                if self.snapshot {
                    Self::run_backfill_snapshot(
                        &provider_factory,
                        storage,
                        self.target_earliest_block,
                    )?;
                } else {
                    Self::run_backfill_merge_walk(
                        &provider_factory,
                        storage,
                        self.target_earliest_block,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Open a long-read-safe reth RO provider for the backfill run, logging
    /// the proofs-window state and target as a side effect.
    fn prepare_backfill_provider<F, S>(
        provider_factory: &F,
        storage: &S,
        target_earliest_block: u64,
        strategy: &'static str,
    ) -> eyre::Result<<F as DatabaseProviderFactory>::Provider>
    where
        F: DatabaseProviderFactory,
        S: OpProofsStore + Send,
    {
        let ro = storage.provider_ro()?;
        let earliest = ro.get_earliest_block_number()?;
        let latest = ro.get_latest_block_number()?;
        drop(ro);
        info!(
            target: "reth::cli",
            strategy,
            ?earliest,
            ?latest,
            target_earliest_block,
            "Starting backfill job"
        );

        Ok(provider_factory
            .database_provider_ro()
            .map_err(|e| eyre::eyre!("Failed to open reth DB provider: {e}"))?
            .disable_long_read_transaction_safety())
    }

    fn run_backfill_merge_walk<F, S>(
        provider_factory: &F,
        storage: S,
        target_earliest_block: u64,
    ) -> eyre::Result<()>
    where
        F: DatabaseProviderFactory,
        F::Provider: DBProvider
            + StageCheckpointReader
            + ChangeSetReader
            + StorageChangeSetReader
            + BlockNumReader
            + BlockHashReader
            + HeaderProvider
            + StorageSettingsCache
            + Send,
        S: OpProofsStore + Send,
    {
        let provider = Self::prepare_backfill_provider(
            provider_factory,
            &storage,
            target_earliest_block,
            "merge-walk",
        )?;
        BackfillJob::new(provider, storage).run(target_earliest_block)?;
        Ok(())
    }

    fn run_backfill_snapshot<F, S>(
        provider_factory: &F,
        storage: S,
        target_earliest_block: u64,
    ) -> eyre::Result<()>
    where
        F: DatabaseProviderFactory,
        F::Provider: DBProvider
            + StageCheckpointReader
            + ChangeSetReader
            + StorageChangeSetReader
            + BlockNumReader
            + BlockHashReader
            + HeaderProvider
            + StorageSettingsCache
            + Send
            + Sync,
        S: OpProofsSnapshotStore + Send,
        for<'a> S::ProviderRO<'a>: OpProofsSnapshotProviderRO,
        for<'a> S::BackfillProvider<'a>: OpProofsSnapshotProviderRW,
    {
        let provider = Self::prepare_backfill_provider(
            provider_factory,
            &storage,
            target_earliest_block,
            "snapshot",
        )?;
        BackfillJob::new(provider, storage).run_auto(target_earliest_block)?;
        Ok(())
    }
}

impl<C: ChainSpecParser> BackfillCommand<C> {
    /// Returns the underlying chain being used to run this command
    pub const fn chain_spec(&self) -> Option<&Arc<C::ChainSpec>> {
        Some(&self.env.chain)
    }
}
