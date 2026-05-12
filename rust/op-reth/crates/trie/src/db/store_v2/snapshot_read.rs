//! [`OpProofsSnapshotProviderRO`] implementation for [`MdbxProofsProviderV2`].

use super::{MdbxProofsProviderV2, cursor::{V2AccountTrieSnapshotCursor, V2StorageTrieSnapshotCursor}};
use crate::{
    OpProofsStorageResult,
    api::OpProofsSnapshotProviderRO,
    db::{
        SnapshotMeta, SnapshotMetaKey,
        models::{V2AccountsTrieSnapshot, V2StoragesTrieSnapshot, V2TrieSnapshotMeta},
    },
};
use alloy_primitives::B256;
use reth_db::{cursor::DbCursorRO, transaction::DbTx};
use std::fmt::Debug;

impl<TX: DbTx + Send + Sync + Debug + 'static> OpProofsSnapshotProviderRO for MdbxProofsProviderV2<TX> {
    type SnapshotAccountTrieCursor<'tx>
        = V2AccountTrieSnapshotCursor<TX::Cursor<V2AccountsTrieSnapshot>>
    where
        Self: 'tx,
        TX: 'tx;

    type SnapshotStorageTrieCursor<'tx>
        = V2StorageTrieSnapshotCursor<TX::DupCursor<V2StoragesTrieSnapshot>>
    where
        Self: 'tx,
        TX: 'tx;

    fn snapshot_meta(&self) -> OpProofsStorageResult<Option<SnapshotMeta>> {
        let mut cursor = self.tx.cursor_read::<V2TrieSnapshotMeta>()?;
        Ok(cursor.seek_exact(SnapshotMetaKey::Singleton)?.map(|(_, meta)| meta))
    }

    fn snapshot_account_trie_cursor<'tx>(
        &self,
    ) -> OpProofsStorageResult<Self::SnapshotAccountTrieCursor<'tx>> {
        Ok(V2AccountTrieSnapshotCursor::new(self.tx.cursor_read::<V2AccountsTrieSnapshot>()?))
    }

    fn snapshot_storage_trie_cursor<'tx>(
        &self,
        hashed_address: B256,
    ) -> OpProofsStorageResult<Self::SnapshotStorageTrieCursor<'tx>> {
        Ok(V2StorageTrieSnapshotCursor::new(
            self.tx.cursor_dup_read::<V2StoragesTrieSnapshot>()?,
            hashed_address,
        ))
    }
}
