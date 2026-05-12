//! Command that builds the V2 trie-state snapshot used by snapshot-mode backfill.

use clap::Parser;
use reth_cli::chainspec::ChainSpecParser;
use reth_cli_commands::common::{AccessRights, CliNodeTypes, Environment, EnvironmentArgs};
use reth_node_core::version::version_metadata;
use reth_optimism_chainspec::OpChainSpec;
use reth_optimism_primitives::OpPrimitives;
use reth_optimism_trie::{
    OpProofsProviderRO, OpProofsStore, SnapshotInitJob,
    db::MdbxProofsStorageV2,
};
use reth_provider::{DBProvider, DatabaseProviderFactory};
use std::{path::PathBuf, sync::Arc};
use tracing::info;

/// Builds the trie-state snapshot at the current `latest` block.
///
/// Required (or auto-built by `backfill --proofs-history.snapshot`) before the
/// snapshot fast-path can run. Refuses to overwrite an existing snapshot —
/// use `snapshot-drop` first to rebuild.
#[derive(Debug, Parser)]
pub struct SnapshotInitCommand<C: ChainSpecParser> {
    #[command(flatten)]
    env: EnvironmentArgs<C>,

    /// The path to the V2 proofs storage DB.
    #[arg(
        long = "proofs-history.storage-path",
        value_name = "PROOFS_HISTORY_STORAGE_PATH",
        required = true
    )]
    pub storage_path: PathBuf,

    /// Block at which to anchor the snapshot. Defaults to the proofs window's
    /// current `earliest` block (the typical case for accelerating subsequent
    /// backfills). Must fall inside `[earliest, latest]`.
    #[arg(long = "proofs-history.target-block", value_name = "TARGET_BLOCK")]
    pub target_block: Option<u64>,
}

impl<C: ChainSpecParser<ChainSpec = OpChainSpec>> SnapshotInitCommand<C> {
    /// Execute `op-proofs snapshot-init`.
    pub async fn execute<N: CliNodeTypes<ChainSpec = C::ChainSpec, Primitives = OpPrimitives>>(
        self,
        runtime: reth_tasks::Runtime,
    ) -> eyre::Result<()> {
        info!(target: "reth::cli", "reth {} starting", version_metadata().short_version);
        info!(
            target: "reth::cli",
            "Building OP proofs V2 snapshot at: {:?}",
            self.storage_path
        );

        let Environment { provider_factory, .. } = self.env.init::<N>(AccessRights::RO, runtime)?;

        let storage: Arc<MdbxProofsStorageV2> = Arc::new(
            MdbxProofsStorageV2::new(&self.storage_path)
                .map_err(|e| eyre::eyre!("Failed to open V2 proofs storage: {e}"))?,
        );

        // Resolve target block: caller-provided or the proofs-window earliest.
        let target_block = match self.target_block {
            Some(b) => b,
            None => storage
                .provider_ro()
                .map_err(|e| eyre::eyre!("Failed to open proofs provider: {e}"))?
                .get_earliest_block_number()?
                .map(|(n, _)| n)
                .ok_or_else(|| {
                    eyre::eyre!("Proofs storage has no earliest block — initialize it first")
                })?,
        };

        let provider = provider_factory
            .database_provider_ro()
            .map_err(|e| eyre::eyre!("Failed to open reth DB provider: {e}"))?
            .disable_long_read_transaction_safety();

        let outcome = SnapshotInitJob::new(provider, storage).run(target_block)?;
        info!(
            target: "reth::cli",
            earliest = outcome.meta.earliest.number,
            account_nodes_copied = outcome.account_nodes_copied,
            storage_nodes_copied = outcome.storage_nodes_copied,
            "Snapshot ready"
        );
        Ok(())
    }
}

impl<C: ChainSpecParser> SnapshotInitCommand<C> {
    /// Returns the underlying chain being used to run this command.
    pub const fn chain_spec(&self) -> Option<&Arc<C::ChainSpec>> {
        Some(&self.env.chain)
    }
}
