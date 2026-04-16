//! V2 MDBX implementation of [`OpProofsStore`].
//!
//! This module implements the v2 table schema using **3-table-per-data-type** pattern:
//!
//! | Domain | Current State | ChangeSet | History Bitmap |
//! |--------|--------------|-----------|----------------|
//! | Hashed Accounts | [`V2HashedAccounts`] | [`V2HashedAccountChangeSets`] | [`V2HashedAccountsHistory`] |
//! | Hashed Storages | [`V2HashedStorages`] | [`V2HashedStorageChangeSets`] | [`V2HashedStoragesHistory`] |
//! | Account Trie | [`V2AccountsTrie`] | [`V2AccountTrieChangeSets`] | [`V2AccountsTrieHistory`] |
//! | Storage Trie | [`V2StoragesTrie`] | [`V2StorageTrieChangeSets`] | [`V2StoragesTrieHistory`] |

pub(crate) mod cursor;
mod init;
mod provider_ro;
mod provider_rw;
mod read;
mod write;

pub use cursor::{V2AccountCursor, V2AccountTrieCursor, V2StorageCursor, V2StorageTrieCursor};

#[cfg(test)]
mod tests;

use super::Tables;
use crate::{OpProofsStorageError, OpProofsStorageResult, api::OpProofsStore};
#[cfg(feature = "metrics")]
use metrics::{Label, gauge};
use reth_db::{
    Database, DatabaseEnv, DatabaseError,
    mdbx::{DatabaseArguments, init_db_for},
};
use std::{path::Path, sync::Arc};

/// Maximum number of block indices per shard in history bitmap tables.
pub(super) const NUM_OF_INDICES_IN_SHARD: usize = 2_000;

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
    pub(super) tx: TX,
}

impl<TX> MdbxProofsProviderV2<TX> {
    /// Creates a new [`MdbxProofsProviderV2`].
    pub const fn new(tx: TX) -> Self {
        Self { tx }
    }
}
