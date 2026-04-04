use alloy_primitives::{BlockNumber, B256};
use reth_db::{
    models::sharded_key::ShardedKey,
    table::{Decode, Encode},
    DatabaseError,
};
use reth_trie_common::StoredNibbles;
use serde::{Deserialize, Serialize};

/// Keys Hashed Storage History by: Hashed Address + Sharded Key (Storage Key + Sharded Block).
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct HashedStorageShardedKey {
    /// The hashed address of the account owning the storage.
    pub hashed_address: B256,
    /// The sharded key combining the storage key and sharded block number.
    pub sharded_key: ShardedKey<B256>,
}

impl Encode for HashedStorageShardedKey {
    type Encoded = Vec<u8>;
    fn encode(self) -> Self::Encoded {
        let mut buf = Vec::with_capacity(32 + 32 + 8);
        buf.extend_from_slice(self.hashed_address.as_slice());
        // ShardedKey<B256>: Key (32 bytes) + BlockNumber (8 bytes BE)
        buf.extend_from_slice(self.sharded_key.key.as_slice());
        buf.extend_from_slice(&self.sharded_key.highest_block_number.to_be_bytes());
        buf
    }
}

impl Decode for HashedStorageShardedKey {
    fn decode(value: &[u8]) -> Result<Self, DatabaseError> {
        // 32 (Addr) + 32 (Key) + 8 (Block) = 72 bytes
        if value.len() < 72 {
            return Err(DatabaseError::Decode);
        }
        let (addr, rest) = value.split_at(32);
        let hashed_address = B256::from_slice(addr);
        let key = B256::from_slice(&rest[..32]);
        let highest_block_number =
            u64::from_be_bytes(rest[32..40].try_into().map_err(|_| DatabaseError::Decode)?);
        Ok(Self {
            hashed_address,
            sharded_key: ShardedKey::new(key, highest_block_number),
        })
    }
}

/// Keys Storage `ChangeSets` by: Block Number + Hashed Address.
/// Replaces `BlockNumberAddress` which uses unhashed Address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct BlockNumberHashedAddress(pub (BlockNumber, B256));

impl Encode for BlockNumberHashedAddress {
    type Encoded = [u8; 40]; // 8 + 32
    fn encode(self) -> Self::Encoded {
        let mut buf = [0u8; 40];
        buf[..8].copy_from_slice(&self.0 .0.to_be_bytes());
        buf[8..].copy_from_slice(self.0 .1.as_slice());
        buf
    }
}

impl Decode for BlockNumberHashedAddress {
    fn decode(value: &[u8]) -> Result<Self, DatabaseError> {
        if value.len() < 40 {
            return Err(DatabaseError::Decode);
        }
        let block_num = u64::from_be_bytes(value[..8].try_into().unwrap());
        let hash = B256::from_slice(&value[8..40]);
        Ok(Self((block_num, hash)))
    }
}

/// Keys Storage Trie History by: Hashed Address + Nibbles + Sharded Block.
///
/// Uses **length-prefixed encoding** for the nibble portion to avoid sort
/// ambiguity in MDBX (same rationale as [`AccountTrieShardedKey`](super::key::AccountTrieShardedKey)):
///
/// ```text
/// [hashed_address: 32 bytes] ++ [nibble_count: 1 byte] ++ [nibble_bytes] ++ [block_number: 8 BE bytes]
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct StorageTrieShardedKey {
    /// The hashed address of the account owning the storage trie.
    pub hashed_address: B256,
    /// The trie path (nibbles).
    pub key: StoredNibbles,
    /// Highest block number in this shard (or `u64::MAX` for the sentinel).
    pub highest_block_number: u64,
}

impl StorageTrieShardedKey {
    /// Create a new storage trie sharded key.
    pub const fn new(hashed_address: B256, key: StoredNibbles, highest_block_number: u64) -> Self {
        Self { hashed_address, key, highest_block_number }
    }
}

impl Encode for StorageTrieShardedKey {
    type Encoded = Vec<u8>;
    fn encode(self) -> Self::Encoded {
        let nibble_bytes: Vec<u8> = self.key.0.iter().collect();
        let nibble_count = nibble_bytes.len() as u8;
        let mut buf = Vec::with_capacity(32 + 1 + nibble_bytes.len() + 8);
        buf.extend_from_slice(self.hashed_address.as_slice());
        buf.push(nibble_count);
        buf.extend_from_slice(&nibble_bytes);
        buf.extend_from_slice(&self.highest_block_number.to_be_bytes());
        buf
    }
}

impl Decode for StorageTrieShardedKey {
    fn decode(value: &[u8]) -> Result<Self, DatabaseError> {
        // Minimum: 32 (addr) + 1 (count) + 0 (nibbles) + 8 (block) = 41
        if value.len() < 41 {
            return Err(DatabaseError::Decode);
        }
        let hashed_address = B256::from_slice(&value[..32]);
        let nibble_count = value[32] as usize;
        let expected_len = 32 + 1 + nibble_count + 8;
        if value.len() != expected_len {
            return Err(DatabaseError::Decode);
        }
        let nibble_bytes = &value[33..33 + nibble_count];
        let key = StoredNibbles::from(
            reth_trie_common::Nibbles::from_nibbles_unchecked(nibble_bytes),
        );
        let block_bytes = &value[33 + nibble_count..];
        let highest_block_number =
            u64::from_be_bytes(block_bytes.try_into().map_err(|_| DatabaseError::Decode)?);
        Ok(Self { hashed_address, key, highest_block_number })
    }
}