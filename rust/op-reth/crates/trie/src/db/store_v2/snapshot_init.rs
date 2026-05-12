//! [`OpProofsSnapshotInitProvider`] implementation for [`MdbxProofsProviderV2`].
//!
//! Mirrors `init.rs`'s role for [`OpProofsInitProvider`](crate::api::OpProofsInitProvider):
//! this module is the one place where all init-time operations on the snapshot
//! tables live — bulk source reads, append-only destination writes, anchor
//! recovery, meta transitions, and `clear`.

use super::MdbxProofsProviderV2;
use crate::{
    OpProofsStorageError, OpProofsStorageResult,
    api::{OpProofsSnapshotInitProvider, SnapshotInitAnchor},
    db::{
        SnapshotMeta, SnapshotMetaKey, SnapshotStatus,
        models::{V2AccountsTrieSnapshot, V2StoragesTrieSnapshot, V2TrieSnapshotMeta},
    },
};
use alloy_eips::BlockNumHash;
use alloy_primitives::B256;
use reth_db::{
    DatabaseError,
    cursor::{DbCursorRO, DbCursorRW, DbDupCursorRW},
    transaction::{DbTx, DbTxMut},
};
use reth_trie::{BranchNodeCompact, StorageTrieEntry, StoredNibbles, StoredNibblesSubKey};
use std::fmt::Debug;

impl<TX: DbTxMut + DbTx + Send + Sync + Debug + 'static> OpProofsSnapshotInitProvider
    for MdbxProofsProviderV2<TX>
{
    fn snapshot_init_anchor(&self) -> OpProofsStorageResult<SnapshotInitAnchor> {
        let meta = self
            .tx
            .cursor_read::<V2TrieSnapshotMeta>()?
            .seek_exact(SnapshotMetaKey::Singleton)?
            .map(|(_, m)| m);

        let last_account_trie_key =
            self.tx.cursor_read::<V2AccountsTrieSnapshot>()?.last()?.map(|(k, _)| k);

        let last_storage_trie_key = self
            .tx
            .cursor_dup_read::<V2StoragesTrieSnapshot>()?
            .last()?
            .map(|(addr, entry)| (addr, entry.nibbles));

        Ok(SnapshotInitAnchor { meta, last_account_trie_key, last_storage_trie_key })
    }

    fn set_snapshot_init_anchor(&self, anchor: BlockNumHash) -> OpProofsStorageResult<()> {
        let mut cur = self.tx.cursor_write::<V2TrieSnapshotMeta>()?;
        // `insert` errors if a row is already present at this key — caller
        // must `clear_snapshot` first if they intend to rebuild.
        cur.insert(
            SnapshotMetaKey::Singleton,
            &SnapshotMeta::new(anchor, SnapshotStatus::Building),
        )?;
        Ok(())
    }

    fn clear_snapshot(&self) -> OpProofsStorageResult<()> {
        self.tx.clear::<V2AccountsTrieSnapshot>()?;
        self.tx.clear::<V2StoragesTrieSnapshot>()?;
        self.tx.clear::<V2TrieSnapshotMeta>()?;
        Ok(())
    }

    fn store_account_trie_snapshot_branches(
        &self,
        entries: Vec<(StoredNibbles, BranchNodeCompact)>,
    ) -> OpProofsStorageResult<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut cur = self.tx.cursor_write::<V2AccountsTrieSnapshot>()?;
        for (key, node) in entries {
            cur.append(key, &node)?;
        }
        Ok(())
    }

    fn store_storage_trie_snapshot_branches(
        &self,
        entries: Vec<(B256, StoredNibblesSubKey, BranchNodeCompact)>,
    ) -> OpProofsStorageResult<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut cur = self.tx.cursor_dup_write::<V2StoragesTrieSnapshot>()?;
        for (addr, subkey, node) in entries {
            cur.append_dup(addr, StorageTrieEntry { nibbles: subkey, node })?;
        }
        Ok(())
    }

    fn commit_snapshot(&self) -> OpProofsStorageResult<()> {
        let mut cur = self.tx.cursor_write::<V2TrieSnapshotMeta>()?;
        let existing = cur.seek_exact(SnapshotMetaKey::Singleton)?.map(|(_, m)| m);
        let meta = match existing {
            Some(m) if m.status == SnapshotStatus::Building => m,
            _ => {
                return Err(OpProofsStorageError::DatabaseError(DatabaseError::Other(
                    "commit_snapshot called without a Building meta row".to_string(),
                )));
            }
        };
        cur.upsert(
            SnapshotMetaKey::Singleton,
            &SnapshotMeta::new(meta.earliest, SnapshotStatus::Ready),
        )?;
        Ok(())
    }

    fn commit(self) -> OpProofsStorageResult<()> {
        self.tx.commit()?;
        Ok(())
    }
}
