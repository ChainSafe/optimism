//! [`OpProofsSnapshotInitProvider`] implementation for [`MdbxProofsProviderV2`].
//!
//! Mirrors `init.rs`'s role for [`OpProofsInitProvider`](crate::api::OpProofsInitProvider):
//! this module is the one place where all init-time operations on the snapshot
//! tables live — bulk source reads, append-only destination writes, anchor
//! recovery, meta transitions, and `clear`.

use super::MdbxProofsProviderV2;
use crate::{
    OpProofsStorageResult,
    api::{OpProofsSnapshotInitProvider, SnapshotInitAnchor},
    db::{
        SnapshotMeta, SnapshotMetaKey,
        models::{
            V2AccountsTrie, V2AccountsTrieSnapshot, V2StoragesTrie, V2StoragesTrieSnapshot,
            V2TrieSnapshotMeta,
        },
    },
};
use alloy_primitives::B256;
use reth_db::{
    cursor::{DbCursorRO, DbCursorRW, DbDupCursorRO, DbDupCursorRW},
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

    fn set_snapshot_meta(&self, meta: SnapshotMeta) -> OpProofsStorageResult<()> {
        let mut cur = self.tx.cursor_write::<V2TrieSnapshotMeta>()?;
        cur.upsert(SnapshotMetaKey::Singleton, &meta)?;
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

    fn account_trie_source_chunk(
        &self,
        resume_after: Option<StoredNibbles>,
        max_entries: usize,
    ) -> OpProofsStorageResult<Vec<(StoredNibbles, BranchNodeCompact)>> {
        if max_entries == 0 {
            return Ok(Vec::new());
        }
        let mut cur = self.tx.cursor_read::<V2AccountsTrie>()?;
        let mut next = match resume_after {
            None => cur.first()?,
            Some(after) => {
                // `seek` returns the first key >= `after`. If it matches
                // exactly we want the next one; otherwise we're already past.
                match cur.seek(after.clone())? {
                    Some((k, _)) if k == after => cur.next()?,
                    other => other,
                }
            }
        };
        let mut out = Vec::with_capacity(max_entries);
        while let Some((k, v)) = next {
            if out.len() >= max_entries {
                break;
            }
            out.push((k, v));
            next = cur.next()?;
        }
        Ok(out)
    }

    fn storage_trie_source_chunk(
        &self,
        resume_after: Option<(B256, StoredNibblesSubKey)>,
        max_entries: usize,
    ) -> OpProofsStorageResult<Vec<(B256, StoredNibblesSubKey, BranchNodeCompact)>> {
        if max_entries == 0 {
            return Ok(Vec::new());
        }
        let mut cur = self.tx.cursor_dup_read::<V2StoragesTrie>()?;
        let mut next = match resume_after {
            None => cur.first()?,
            Some((addr, subkey)) => {
                let positioned = cur
                    .seek_by_key_subkey(addr, subkey.clone())?
                    .map(|entry| (addr, entry));
                match positioned {
                    Some((a, e)) if a == addr && e.nibbles == subkey => cur.next()?,
                    Some((a, e)) => Some((a, e)),
                    None => None,
                }
            }
        };
        let mut out = Vec::with_capacity(max_entries);
        while let Some((addr, entry)) = next {
            if out.len() >= max_entries {
                break;
            }
            out.push((addr, entry.nibbles, entry.node));
            next = cur.next()?;
        }
        Ok(out)
    }

    fn commit(self) -> OpProofsStorageResult<()> {
        self.tx.commit()?;
        Ok(())
    }
}
