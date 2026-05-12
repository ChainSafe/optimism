//! [`OpProofsSnapshotProvider`] implementation for [`MdbxProofsProviderV2`].
//!
//! Per-iteration backfill writer surface. The init-time writer
//! ([`OpProofsSnapshotInitProvider`](crate::api::OpProofsSnapshotInitProvider))
//! lives in [`super::snapshot_init`].

use super::MdbxProofsProviderV2;
use crate::{
    OpProofsStorageResult,
    api::OpProofsSnapshotProvider,
    db::{
        SnapshotMeta, SnapshotMetaKey,
        models::{V2AccountsTrieSnapshot, V2StoragesTrieSnapshot, V2TrieSnapshotMeta},
    },
};
use reth_db::{
    cursor::{DbCursorRO, DbCursorRW, DbDupCursorRO},
    transaction::{DbTx, DbTxMut},
};
use reth_trie::{
    StorageTrieEntry, StoredNibbles, StoredNibblesSubKey, updates::TrieUpdatesSorted,
};
use std::fmt::Debug;

impl<TX: DbTxMut + DbTx + Send + Sync + Debug + 'static> OpProofsSnapshotProvider
    for MdbxProofsProviderV2<TX>
{
    fn set_snapshot_meta(&self, meta: SnapshotMeta) -> OpProofsStorageResult<()> {
        let mut cur = self.tx.cursor_write::<V2TrieSnapshotMeta>()?;
        cur.upsert(SnapshotMetaKey::Singleton, &meta)?;
        Ok(())
    }

    fn apply_snapshot_revert(
        &self,
        trie_updates: &TrieUpdatesSorted,
    ) -> OpProofsStorageResult<u64> {
        let mut count = 0u64;

        // Account trie revert.
        let mut acc = self.tx.cursor_write::<V2AccountsTrieSnapshot>()?;
        for (nibbles, maybe_node) in trie_updates.account_nodes_ref() {
            let key = StoredNibbles(*nibbles);
            match maybe_node {
                Some(node) => acc.upsert(key, node)?,
                None => {
                    if acc.seek_exact(key)?.is_some() {
                        acc.delete_current()?;
                    }
                }
            }
            count += 1;
        }

        // Storage trie revert.
        let mut stor = self.tx.cursor_dup_write::<V2StoragesTrieSnapshot>()?;
        for (hashed_address, nodes) in trie_updates.storage_tries_ref() {
            for (nibbles, maybe_node) in nodes.storage_nodes_ref() {
                let subkey = StoredNibblesSubKey(*nibbles);
                let existing = stor
                    .seek_by_key_subkey(*hashed_address, subkey.clone())?
                    .filter(|e| e.nibbles == subkey)
                    .is_some();
                if existing {
                    stor.delete_current()?;
                }
                if let Some(node) = maybe_node {
                    stor.upsert(
                        *hashed_address,
                        &StorageTrieEntry { nibbles: subkey, node: node.clone() },
                    )?;
                }
                count += 1;
            }
        }

        Ok(count)
    }

    fn commit(self) -> OpProofsStorageResult<()> {
        self.tx.commit()?;
        Ok(())
    }
}
