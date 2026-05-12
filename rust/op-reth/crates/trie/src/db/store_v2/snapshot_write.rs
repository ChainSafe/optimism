//! [`OpProofsSnapshotProviderRW`] implementation for [`MdbxProofsProviderV2`].
//!
//! Per-iteration backfill writer surface. The init-time writer
//! ([`OpProofsSnapshotInitProvider`](crate::api::OpProofsSnapshotInitProvider))
//! lives in [`super::snapshot_init`].

use super::MdbxProofsProviderV2;
use crate::{
    OpProofsStorageError, OpProofsStorageResult,
    api::OpProofsSnapshotProviderRW,
    db::{
        SnapshotMeta, SnapshotMetaKey, SnapshotStatus,
        models::{V2AccountsTrieSnapshot, V2StoragesTrieSnapshot, V2TrieSnapshotMeta},
    },
};
use alloy_eips::BlockNumHash;
use reth_db::{
    DatabaseError,
    cursor::{DbCursorRO, DbCursorRW, DbDupCursorRO},
    transaction::{DbTx, DbTxMut},
};
use reth_trie::{
    StorageTrieEntry, StoredNibbles, StoredNibblesSubKey, updates::TrieUpdatesSorted,
};
use std::fmt::Debug;

impl<TX: DbTxMut + DbTx + Send + Sync + Debug + 'static> OpProofsSnapshotProviderRW
    for MdbxProofsProviderV2<TX>
{
    fn update_snapshot(
        &self,
        new_anchor: BlockNumHash,
        trie_updates: &TrieUpdatesSorted,
    ) -> OpProofsStorageResult<u64> {
        // Refuse to advance a snapshot that's still being built — the partial
        // tables under `Building` aren't a valid anchor for a Ready move.
        if let Some((_, meta)) = self
            .tx
            .cursor_read::<V2TrieSnapshotMeta>()?
            .seek_exact(SnapshotMetaKey::Singleton)? &&
            meta.status == SnapshotStatus::Building
        {
            return Err(OpProofsStorageError::DatabaseError(DatabaseError::Other(format!(
                "update_snapshot called on a Building snapshot at block {}",
                meta.earliest.number
            ))));
        }

        let mut count = 0u64;

        // Account trie diff.
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

        // Storage trie diff.
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

        // Advance the anchor atomically with the data write.
        let mut meta_cur = self.tx.cursor_write::<V2TrieSnapshotMeta>()?;
        meta_cur.upsert(
            SnapshotMetaKey::Singleton,
            &SnapshotMeta::new(new_anchor, SnapshotStatus::Ready),
        )?;

        Ok(count)
    }

    fn commit(self) -> OpProofsStorageResult<()> {
        self.tx.commit()?;
        Ok(())
    }
}
