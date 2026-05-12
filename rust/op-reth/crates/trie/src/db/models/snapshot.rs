//! Models for the backfill trie-state snapshot (see
//! [`crate::backfill`] for the design rationale).

use alloy_eips::BlockNumHash;
use alloy_primitives::B256;
use bytes::BufMut;
use reth_codecs::DecompressError;
use reth_db::{
    DatabaseError,
    table::{Compress, Decode, Decompress, Encode},
};
use serde::{Deserialize, Serialize};

/// Single-row key for the snapshot metadata table.
///
/// There is only ever one snapshot per proofs store, so the table has a
/// fixed singleton key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum SnapshotMetaKey {
    /// The singleton key — there is only ever one snapshot meta row.
    Singleton = 0,
}

impl Encode for SnapshotMetaKey {
    type Encoded = [u8; 1];

    fn encode(self) -> Self::Encoded {
        [self as u8]
    }
}

impl Decode for SnapshotMetaKey {
    fn decode(value: &[u8]) -> Result<Self, DatabaseError> {
        match value.first() {
            Some(&0) => Ok(Self::Singleton),
            _ => Err(DatabaseError::Decode),
        }
    }
}

/// Lifecycle status of the trie snapshot.
///
/// - [`Self::Building`]: a [`crate::backfill::SnapshotInitJob`] is populating
///   the snapshot tables. Reads against the snapshot must be refused.
/// - [`Self::Ready`]: the snapshot reflects valid trie state at
///   [`SnapshotMeta::earliest`]. Backfill can use it.
/// - [`Self::Stale`]: the snapshot exists but no longer matches the current
///   `earliest` (e.g. a previous backfill ran to completion, advancing
///   `earliest` past where the snapshot tracks). Reads must be refused;
///   either rebuild or drop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum SnapshotStatus {
    /// Snapshot is being constructed by [`crate::backfill::SnapshotInitJob`].
    /// Reads must be refused until status transitions to [`Self::Ready`].
    Building = 0,
    /// Snapshot is consistent and reflects trie state at [`SnapshotMeta::earliest`].
    Ready = 1,
    /// Snapshot was once Ready but the proofs window has moved past where it
    /// tracks (e.g. backfill ran to completion). Must be rebuilt before reuse.
    Stale = 2,
}

/// Metadata for the trie-state snapshot.
///
/// Encoding: `[status: 1B] ‖ [block_number: 8B BE] ‖ [block_hash: 32B]` (= 41 B).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotMeta {
    /// The block (number + hash) the snapshot's trie state corresponds to.
    pub earliest: BlockNumHash,
    /// Current lifecycle state.
    pub status: SnapshotStatus,
}

impl SnapshotMeta {
    /// Encoded byte length.
    pub const ENCODED_LEN: usize = 1 + 8 + 32;

    /// Convenience constructor.
    pub const fn new(earliest: BlockNumHash, status: SnapshotStatus) -> Self {
        Self { earliest, status }
    }
}

impl Compress for SnapshotMeta {
    type Compressed = Vec<u8>;

    fn compress_to_buf<B: BufMut + AsMut<[u8]>>(&self, buf: &mut B) {
        buf.put_u8(self.status as u8);
        buf.put_u64(self.earliest.number);
        buf.put_slice(self.earliest.hash.as_slice());
    }
}

impl Decompress for SnapshotMeta {
    fn decompress(value: &[u8]) -> Result<Self, DecompressError> {
        if value.len() != Self::ENCODED_LEN {
            return Err(DecompressError::new(DatabaseError::Decode));
        }
        let status = match value[0] {
            0 => SnapshotStatus::Building,
            1 => SnapshotStatus::Ready,
            2 => SnapshotStatus::Stale,
            _ => return Err(DecompressError::new(DatabaseError::Decode)),
        };
        let number = u64::from_be_bytes(
            value[1..9].try_into().map_err(|_| DecompressError::new(DatabaseError::Decode))?,
        );
        let hash = B256::from_slice(&value[9..41]);
        Ok(Self { earliest: BlockNumHash::new(number, hash), status })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_meta_key_roundtrip() {
        let encoded = SnapshotMetaKey::Singleton.encode();
        let decoded = SnapshotMetaKey::decode(&encoded).unwrap();
        assert_eq!(decoded, SnapshotMetaKey::Singleton);
    }

    #[test]
    fn snapshot_meta_roundtrip_all_statuses() {
        let earliest = BlockNumHash::new(12_345_678, B256::repeat_byte(0xab));
        for status in [SnapshotStatus::Building, SnapshotStatus::Ready, SnapshotStatus::Stale] {
            let original = SnapshotMeta::new(earliest, status);
            let compressed = original.compress();
            assert_eq!(compressed.len(), SnapshotMeta::ENCODED_LEN);
            let decompressed = SnapshotMeta::decompress(&compressed).unwrap();
            assert_eq!(original, decompressed);
        }
    }

    #[test]
    fn snapshot_meta_decompress_rejects_wrong_length() {
        assert!(SnapshotMeta::decompress(&[0u8; 10]).is_err());
        assert!(SnapshotMeta::decompress(&[0u8; 41 + 1]).is_err());
    }

    #[test]
    fn snapshot_meta_decompress_rejects_invalid_status() {
        let mut buf = vec![0xff_u8; SnapshotMeta::ENCODED_LEN];
        // status byte at position 0; 0xff is not a valid SnapshotStatus.
        buf[0] = 0xff;
        assert!(SnapshotMeta::decompress(&buf).is_err());
    }
}
