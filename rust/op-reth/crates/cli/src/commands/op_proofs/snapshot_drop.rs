//! Command that drops the V2 trie-state snapshot tables (and meta).
//!
//! Use this when you want to free disk space taken by a snapshot you no
//! longer need, or to force a rebuild on the next snapshot-mode backfill.

use clap::Parser;
use reth_cli::chainspec::ChainSpecParser;
use reth_cli_commands::common::{AccessRights, CliNodeTypes, EnvironmentArgs};
use reth_node_core::version::version_metadata;
use reth_optimism_chainspec::OpChainSpec;
use reth_optimism_primitives::OpPrimitives;
use reth_optimism_trie::{
    OpProofsSnapshotInitProvider, OpProofsSnapshotReader, OpProofsStore,
    db::MdbxProofsStorageV2,
};
use std::{path::PathBuf, sync::Arc};
use tracing::info;

/// Drops the V2 trie-state snapshot (clears tables and meta).
///
/// Safe to run even if no snapshot exists — in that case it's a no-op.
#[derive(Debug, Parser)]
pub struct SnapshotDropCommand<C: ChainSpecParser> {
    #[command(flatten)]
    env: EnvironmentArgs<C>,

    /// The path to the V2 proofs storage DB.
    #[arg(
        long = "proofs-history.storage-path",
        value_name = "PROOFS_HISTORY_STORAGE_PATH",
        required = true
    )]
    pub storage_path: PathBuf,
}

impl<C: ChainSpecParser<ChainSpec = OpChainSpec>> SnapshotDropCommand<C> {
    /// Execute `op-proofs snapshot-drop`.
    pub async fn execute<N: CliNodeTypes<ChainSpec = C::ChainSpec, Primitives = OpPrimitives>>(
        self,
        runtime: reth_tasks::Runtime,
    ) -> eyre::Result<()> {
        info!(target: "reth::cli", "reth {} starting", version_metadata().short_version);
        info!(
            target: "reth::cli",
            "Dropping OP proofs V2 snapshot at: {:?}",
            self.storage_path
        );

        // Init the environment so the runtime is wired up even though we don't
        // touch the reth DB. Mirrors the other op-proofs subcommands.
        let _env = self.env.init::<N>(AccessRights::RO, runtime)?;

        let storage: Arc<MdbxProofsStorageV2> = Arc::new(
            MdbxProofsStorageV2::new(&self.storage_path)
                .map_err(|e| eyre::eyre!("Failed to open V2 proofs storage: {e}"))?,
        );

        // Log the prior state so the operator sees what's being thrown away.
        let prior = storage.provider_ro()?.snapshot_meta()?;
        match prior {
            Some(meta) => info!(
                target: "reth::cli",
                earliest = meta.earliest.number,
                status = ?meta.status,
                "Existing snapshot — clearing"
            ),
            None => info!(target: "reth::cli", "No snapshot present — clear is a no-op"),
        }

        let bp = storage.backfill_provider()?;
        bp.clear_snapshot()?;
        OpProofsSnapshotInitProvider::commit(bp)?;

        info!(target: "reth::cli", "Snapshot dropped");
        Ok(())
    }
}

impl<C: ChainSpecParser> SnapshotDropCommand<C> {
    /// Returns the underlying chain being used to run this command.
    pub const fn chain_spec(&self) -> Option<&Arc<C::ChainSpec>> {
        Some(&self.env.chain)
    }
}
