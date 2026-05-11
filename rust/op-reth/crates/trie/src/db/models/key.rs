use alloy_primitives::{B256, BlockNumber};
use reth_db::{
    DatabaseError,
    models::sharded_key::ShardedKey,
    table::{Decode, Encode},
};
use reth_trie_common::StoredNibbles;
use serde::{Deserialize, Serialize};

/// Sharded key for hashed accounts history.
///
/// Wraps `ShardedKey<B256>` to provide `Encode`/`Decode` impls needed by MDBX.
#[derive(Debug, Default, Clone, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize, Hash)]
pub struct HashedAccountShardedKey(pub ShardedKey<B256>);

impl HashedAccountShardedKey {
    /// Create a new sharded key for a hashed account.
    pub const fn new(key: B256, highest_block_number: u64) -> Self {
        Self(ShardedKey::new(key, highest_block_number))
    }
}

impl Encode for HashedAccountShardedKey {
    type Encoded = [u8; 40]; // 32 (B256) + 8 (BlockNumber)

    fn encode(self) -> Self::Encoded {
        let mut buf = [0u8; 40];
        buf[..32].copy_from_slice(self.0.key.as_slice());
        buf[32..].copy_from_slice(&self.0.highest_block_number.to_be_bytes());
        buf
    }
}

impl Decode for HashedAccountShardedKey {
    fn decode(value: &[u8]) -> Result<Self, DatabaseError> {
        if value.len() != 40 {
            return Err(DatabaseError::Decode);
        }
        let key = B256::from_slice(&value[..32]);
        let highest_block_number =
            u64::from_be_bytes(value[32..].try_into().map_err(|_| DatabaseError::Decode)?);
        Ok(Self(ShardedKey::new(key, highest_block_number)))
    }
}

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
        Ok(Self { hashed_address, sharded_key: ShardedKey::new(key, highest_block_number) })
    }
}

/// Maximum trie-path length in nibbles (32 bytes of hashed address/slot = 64 nibbles).
const MAX_NIBBLES: usize = 64;
/// Byte length of an encoded [`AccountTrieShardedKey`]: 64 padded nibbles + 1 length + 8 block.
const ACCOUNT_TRIE_SHARDED_KEY_LEN: usize = MAX_NIBBLES + 1 + 8;
/// Byte length of an encoded [`StorageTrieShardedKey`]: 32 addr + 64 padded nibbles + 1 length + 8
/// block.
const STORAGE_TRIE_SHARDED_KEY_LEN: usize = 32 + MAX_NIBBLES + 1 + 8;

/// Sharded key for account trie history.
///
/// Fixed-size encoding chosen so MDBX's byte-wise sort matches `Nibbles`'
/// logical (lex-by-nibble) order:
///
/// ```text
/// [nibbles: 64 bytes, right-padded with 0x00] ++ [length: 1 byte] ++ [block: 8 BE bytes]
/// ```
#[derive(Debug, Default, Clone, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize, Hash)]
pub struct AccountTrieShardedKey {
    /// Trie path as nibbles.
    pub key: StoredNibbles,
    /// Highest block number in this shard (or `u64::MAX` for the sentinel).
    pub highest_block_number: u64,
}

impl AccountTrieShardedKey {
    /// Create a new sharded key for an account trie path.
    pub const fn new(key: StoredNibbles, highest_block_number: u64) -> Self {
        Self { key, highest_block_number }
    }
}

impl Encode for AccountTrieShardedKey {
    type Encoded = [u8; ACCOUNT_TRIE_SHARDED_KEY_LEN];

    fn encode(self) -> Self::Encoded {
        let nibble_bytes: Vec<u8> = self.key.0.iter().collect();
        debug_assert!(
            nibble_bytes.len() <= MAX_NIBBLES,
            "nibble path exceeds {MAX_NIBBLES}: got {}",
            nibble_bytes.len()
        );
        let mut buf = [0u8; ACCOUNT_TRIE_SHARDED_KEY_LEN];
        buf[..nibble_bytes.len()].copy_from_slice(&nibble_bytes);
        buf[MAX_NIBBLES] = nibble_bytes.len() as u8;
        buf[MAX_NIBBLES + 1..].copy_from_slice(&self.highest_block_number.to_be_bytes());
        buf
    }
}

impl Decode for AccountTrieShardedKey {
    fn decode(value: &[u8]) -> Result<Self, DatabaseError> {
        if value.len() != ACCOUNT_TRIE_SHARDED_KEY_LEN {
            return Err(DatabaseError::Decode);
        }
        let nibble_count = value[MAX_NIBBLES] as usize;
        if nibble_count > MAX_NIBBLES {
            return Err(DatabaseError::Decode);
        }
        let key = StoredNibbles::from(reth_trie_common::Nibbles::from_nibbles_unchecked(
            &value[..nibble_count],
        ));
        let highest_block_number = u64::from_be_bytes(
            value[MAX_NIBBLES + 1..].try_into().map_err(|_| DatabaseError::Decode)?,
        );
        Ok(Self { key, highest_block_number })
    }
}

/// Keys Storage Trie History by: Hashed Address + Nibbles + Sharded Block.
///
/// Uses the same right-padded + length-suffix scheme as
/// [`AccountTrieShardedKey`],
/// prefixed by the 32-byte hashed address:
///
/// ```text
/// [hashed_address: 32 bytes] ++ [nibbles: 64 bytes, right-padded with 0x00] ++ [length: 1 byte] ++ [block: 8 BE bytes]
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
    type Encoded = [u8; STORAGE_TRIE_SHARDED_KEY_LEN];

    fn encode(self) -> Self::Encoded {
        let nibble_bytes: Vec<u8> = self.key.0.iter().collect();
        debug_assert!(
            nibble_bytes.len() <= MAX_NIBBLES,
            "nibble path exceeds {MAX_NIBBLES}: got {}",
            nibble_bytes.len()
        );
        let mut buf = [0u8; STORAGE_TRIE_SHARDED_KEY_LEN];
        buf[..32].copy_from_slice(self.hashed_address.as_slice());
        buf[32..32 + nibble_bytes.len()].copy_from_slice(&nibble_bytes);
        buf[32 + MAX_NIBBLES] = nibble_bytes.len() as u8;
        buf[32 + MAX_NIBBLES + 1..].copy_from_slice(&self.highest_block_number.to_be_bytes());
        buf
    }
}

impl Decode for StorageTrieShardedKey {
    fn decode(value: &[u8]) -> Result<Self, DatabaseError> {
        if value.len() != STORAGE_TRIE_SHARDED_KEY_LEN {
            return Err(DatabaseError::Decode);
        }
        let hashed_address = B256::from_slice(&value[..32]);
        let nibble_count = value[32 + MAX_NIBBLES] as usize;
        if nibble_count > MAX_NIBBLES {
            return Err(DatabaseError::Decode);
        }
        let key = StoredNibbles::from(reth_trie_common::Nibbles::from_nibbles_unchecked(
            &value[32..32 + nibble_count],
        ));
        let highest_block_number = u64::from_be_bytes(
            value[32 + MAX_NIBBLES + 1..].try_into().map_err(|_| DatabaseError::Decode)?,
        );
        Ok(Self { hashed_address, key, highest_block_number })
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
        buf[..8].copy_from_slice(&self.0.0.to_be_bytes());
        buf[8..].copy_from_slice(self.0.1.as_slice());
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

#[cfg(test)]
mod tests {
    use super::*;
    use reth_db::table::{Decode, Encode};
    use reth_trie_common::Nibbles;

    #[test]
    fn hashed_account_sharded_key_roundtrip() {
        let original = HashedAccountShardedKey::new(B256::repeat_byte(0xaa), 42);
        let decoded = HashedAccountShardedKey::decode(&original.clone().encode()).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn hashed_storage_sharded_key_roundtrip() {
        let original = HashedStorageShardedKey {
            hashed_address: B256::repeat_byte(0xaa),
            sharded_key: ShardedKey::new(B256::repeat_byte(0xbb), 100),
        };
        let decoded = HashedStorageShardedKey::decode(&original.clone().encode()).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn account_trie_sharded_key_roundtrip() {
        let nibbles = StoredNibbles::from(Nibbles::from_nibbles_unchecked([0x0a, 0x0b, 0x0c]));
        let original = AccountTrieShardedKey::new(nibbles, 500);
        let decoded = AccountTrieShardedKey::decode(&original.clone().encode()).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn account_trie_sharded_key_roundtrip_empty_nibbles() {
        let original =
            AccountTrieShardedKey::new(StoredNibbles::from(Nibbles::default()), u64::MAX);
        let decoded = AccountTrieShardedKey::decode(&original.clone().encode()).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn storage_trie_sharded_key_roundtrip() {
        let nibbles = StoredNibbles::from(Nibbles::from_nibbles_unchecked([0x01, 0x02]));
        let original = StorageTrieShardedKey::new(B256::repeat_byte(0xcc), nibbles, 999);
        let decoded = StorageTrieShardedKey::decode(&original.clone().encode()).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn storage_trie_sharded_key_roundtrip_empty_nibbles() {
        let original = StorageTrieShardedKey::new(
            B256::repeat_byte(0xdd),
            StoredNibbles::from(Nibbles::default()),
            0,
        );
        let decoded = StorageTrieShardedKey::decode(&original.clone().encode()).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn block_number_hashed_address_roundtrip() {
        let original = BlockNumberHashedAddress((42, B256::repeat_byte(0xdd)));
        let decoded = BlockNumberHashedAddress::decode(&original.encode()).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn account_trie_sort_matches_nibbles_lex_order() {
        let short_high = AccountTrieShardedKey::new(
            StoredNibbles::from(Nibbles::from_nibbles_unchecked([0x0d, 0x0e, 0x0b])),
            u64::MAX,
        );
        let long_low = AccountTrieShardedKey::new(
            StoredNibbles::from(Nibbles::from_nibbles_unchecked([0x01, 0x08, 0x0c, 0x07, 0x08])),
            u64::MAX,
        );
        // Sanity: lex on nibbles puts long_low (0x18c78) before short_high (0xdeb).
        assert!(long_low.key < short_high.key);
        // Encoded byte order must agree.
        assert!(
            long_low.clone().encode() < short_high.clone().encode(),
            "encoded sort must follow Nibbles lex order, not nibble-length-then-bytes"
        );
    }

    #[test]
    fn storage_trie_sort_matches_nibbles_lex_order() {
        let addr = B256::repeat_byte(0xf3);
        let short_high = StorageTrieShardedKey::new(
            addr,
            StoredNibbles::from(Nibbles::from_nibbles_unchecked([0x0d, 0x0e, 0x0b])),
            u64::MAX,
        );
        let long_low = StorageTrieShardedKey::new(
            addr,
            StoredNibbles::from(Nibbles::from_nibbles_unchecked([0x01, 0x08, 0x0c, 0x07, 0x08])),
            u64::MAX,
        );
        assert!(long_low.key < short_high.key);
        assert!(
            long_low.clone().encode() < short_high.clone().encode(),
            "encoded sort must follow Nibbles lex order, not nibble-length-then-bytes"
        );
    }

    #[test]
    fn storage_trie_seek_invariant_via_encoded_order() {
        let addr = B256::repeat_byte(0xf3);

        let seek_input =
            StoredNibbles::from(Nibbles::from_nibbles_unchecked([0x0d, 0x0e, 0x0b, 0x06]));
        let logically_smaller_but_longer =
            StoredNibbles::from(Nibbles::from_nibbles_unchecked([0x01, 0x08, 0x0c, 0x07, 0x08]));
        let logically_smaller_and_shorter =
            StoredNibbles::from(Nibbles::from_nibbles_unchecked([0x0d, 0x0e, 0x0b]));

        // Both candidate entries are LESS than the seek input in Nibbles order.
        assert!(logically_smaller_but_longer < seek_input);
        assert!(logically_smaller_and_shorter < seek_input);

        let seek_input_encoded = StorageTrieShardedKey::new(addr, seek_input, 0).encode();
        let smaller_longer_encoded =
            StorageTrieShardedKey::new(addr, logically_smaller_but_longer, u64::MAX).encode();
        let smaller_shorter_encoded =
            StorageTrieShardedKey::new(addr, logically_smaller_and_shorter, u64::MAX).encode();

        assert!(
            smaller_longer_encoded < seek_input_encoded,
            "longer-but-smaller-nibbles entry must encode less than seek input"
        );
        assert!(
            smaller_shorter_encoded < seek_input_encoded,
            "shorter-and-smaller-nibbles entry must encode less than seek input"
        );
    }

    #[test]
    fn account_trie_shorter_nibbles_sort_before_longer() {
        let key_a = AccountTrieShardedKey::new(
            StoredNibbles::from(Nibbles::from_nibbles_unchecked([0x01])),
            256,
        );
        let key_b = AccountTrieShardedKey::new(
            StoredNibbles::from(Nibbles::from_nibbles_unchecked([0x01, 0x00])),
            1,
        );

        assert!(
            key_a.encode() < key_b.encode(),
            "shorter nibble path must sort before longer in encoded form"
        );
    }

    #[test]
    fn account_trie_same_nibbles_ordered_by_block() {
        let nibbles = StoredNibbles::from(Nibbles::from_nibbles_unchecked([0x0a, 0x0b]));

        let lo = AccountTrieShardedKey::new(nibbles.clone(), 10);
        let hi = AccountTrieShardedKey::new(nibbles, 20);

        assert!(
            lo.encode() < hi.encode(),
            "same nibbles: lower block must sort before higher block"
        );
    }

    #[test]
    fn account_trie_nibbles_resembling_block_bytes_are_unambiguous() {
        let key_a = AccountTrieShardedKey::new(
            StoredNibbles::from(Nibbles::from_nibbles_unchecked([
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05,
            ])),
            1,
        );
        let key_b = AccountTrieShardedKey::new(StoredNibbles::from(Nibbles::default()), 5);

        let enc_a = key_a.encode();
        let enc_b = key_b.encode();
        assert_ne!(enc_a, enc_b, "different logical keys must never produce identical encodings");
        // Empty nibbles (length 0) must sort before non-empty nibbles (length 8).
        assert!(enc_b < enc_a, "empty nibbles must sort before non-empty nibbles");
    }

    #[test]
    fn storage_trie_shorter_nibbles_sort_before_longer() {
        let addr = B256::repeat_byte(0x11);

        let key_a = StorageTrieShardedKey::new(
            addr,
            StoredNibbles::from(Nibbles::from_nibbles_unchecked([0x0f])),
            256,
        );
        let key_b = StorageTrieShardedKey::new(
            addr,
            StoredNibbles::from(Nibbles::from_nibbles_unchecked([0x0f, 0x00])),
            1,
        );

        assert!(
            key_a.encode() < key_b.encode(),
            "shorter nibble path must sort before longer in encoded form"
        );
    }

    #[test]
    fn storage_trie_same_nibbles_ordered_by_block() {
        let addr = B256::repeat_byte(0x22);
        let nibbles = StoredNibbles::from(Nibbles::from_nibbles_unchecked([0x0a]));

        let lo = StorageTrieShardedKey::new(addr, nibbles.clone(), 10);
        let hi = StorageTrieShardedKey::new(addr, nibbles, 20);

        assert!(
            lo.encode() < hi.encode(),
            "same nibbles: lower block must sort before higher block"
        );
    }

    #[test]
    fn storage_trie_nibbles_resembling_block_bytes_are_unambiguous() {
        let addr = B256::repeat_byte(0x33);

        let key_a = StorageTrieShardedKey::new(
            addr,
            StoredNibbles::from(Nibbles::from_nibbles_unchecked([
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05,
            ])),
            1,
        );
        let key_b = StorageTrieShardedKey::new(addr, StoredNibbles::from(Nibbles::default()), 5);

        let enc_a = key_a.encode();
        let enc_b = key_b.encode();
        assert_ne!(enc_a, enc_b, "different logical keys must never produce identical encodings");
        assert!(enc_b < enc_a, "empty nibbles must sort before non-empty nibbles");
    }
}
