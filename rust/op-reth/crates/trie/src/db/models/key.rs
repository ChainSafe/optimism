use alloy_primitives::B256;
use bytes::BufMut;
use reth_codecs::Compact;
use reth_db::{
    models::sharded_key::ShardedKey,
    table::{Compress, Decode, Decompress, Encode},
    DatabaseError,
};
use reth_primitives_traits::{Account, ValueWithSubKey};
use reth_trie_common::{BranchNodeCompact, StoredNibbles, StoredNibblesSubKey};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// HashedAccountShardedKey – newtype over ShardedKey<B256>
// ---------------------------------------------------------------------------
// Upstream reth only provides Encode/Decode for ShardedKey<Address>.
// We need ShardedKey<B256> for HashedAccountsHistory, so we wrap it.

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

// ---------------------------------------------------------------------------
// AccountTrieShardedKey – newtype over ShardedKey<StoredNibbles>
// ---------------------------------------------------------------------------

/// Sharded key for account trie history.
///
/// Wraps `ShardedKey<StoredNibbles>` to provide `Encode`/`Decode` impls needed by MDBX.
#[derive(Debug, Default, Clone, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize, Hash)]
pub struct AccountTrieShardedKey(pub ShardedKey<StoredNibbles>);

impl AccountTrieShardedKey {
    /// Create a new sharded key for an account trie path.
    pub fn new(key: StoredNibbles, highest_block_number: u64) -> Self {
        Self(ShardedKey::new(key, highest_block_number))
    }
}

impl Encode for AccountTrieShardedKey {
    type Encoded = Vec<u8>;

    fn encode(self) -> Self::Encoded {
        let nibble_bytes: Vec<u8> = self.0.key.0.iter().collect();
        let mut buf = Vec::with_capacity(nibble_bytes.len() + 8);
        buf.extend_from_slice(&nibble_bytes);
        buf.extend_from_slice(&self.0.highest_block_number.to_be_bytes());
        buf
    }
}

impl Decode for AccountTrieShardedKey {
    fn decode(value: &[u8]) -> Result<Self, DatabaseError> {
        if value.len() < 8 {
            return Err(DatabaseError::Decode);
        }
        let (nibble_bytes, block_bytes) = value.split_at(value.len() - 8);
        let key = StoredNibbles::from(
            reth_trie_common::Nibbles::from_nibbles_unchecked(nibble_bytes),
        );
        let highest_block_number =
            u64::from_be_bytes(block_bytes.try_into().map_err(|_| DatabaseError::Decode)?);
        Ok(Self(ShardedKey::new(key, highest_block_number)))
    }
}

/// Account state before a block, keyed by hashed address.
///
/// This is the hashed-address equivalent of reth's [`AccountBeforeTx`](reth_db_models::AccountBeforeTx),
/// designed for our v2 `AccountChangeSets` table where keys are `keccak256(address)`.
///
/// Layout: `[hashed_address: 32 bytes][account: Compact-encoded or empty]`
///
/// - The 32-byte hashed address acts as the [`DupSort::SubKey`].
/// - An empty remainder means the account did not exist before this block (creation).
/// - A non-empty remainder is the [`Account`] state before the block was applied.
///
/// [`DupSort::SubKey`]: reth_db::table::DupSort::SubKey
#[derive(Debug, Default, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct HashedAccountBeforeTx {
    /// Hashed address (`keccak256(address)`). Acts as `DupSort::SubKey`.
    pub hashed_address: B256,
    /// Account state before the block. `None` means the account didn't exist.
    pub info: Option<Account>,
}

impl HashedAccountBeforeTx {
    /// Creates a new instance.
    pub const fn new(hashed_address: B256, info: Option<Account>) -> Self {
        Self { hashed_address, info }
    }
}

impl ValueWithSubKey for HashedAccountBeforeTx {
    type SubKey = B256;

    fn get_subkey(&self) -> Self::SubKey {
        self.hashed_address
    }
}

// NOTE: We manually encode subkey + compress the value separately.
// If we compress the entire value (including the SubKey), MDBX's
// `seek_by_key_subkey` won't be able to locate the subkey prefix.
// This mirrors how reth's `AccountBeforeTx` implements `Compact`.
impl Compress for HashedAccountBeforeTx {
    type Compressed = Vec<u8>;

    fn compress_to_buf<B: BufMut + AsMut<[u8]>>(&self, buf: &mut B) {
        // SubKey: raw 32 bytes (uncompressed so MDBX can seek by it)
        buf.put_slice(self.hashed_address.as_slice());
        // Value: compress the account if present, otherwise write nothing
        if let Some(account) = &self.info {
            account.compress_to_buf(buf);
        }
    }
}

impl Decompress for HashedAccountBeforeTx {
    fn decompress(value: &[u8]) -> Result<Self, DatabaseError> {
        if value.len() < 32 {
            return Err(DatabaseError::Decode);
        }

        let hashed_address = B256::from_slice(&value[..32]);
        let info = if value.len() > 32 {
            Some(Account::decompress(&value[32..])?)
        } else {
            None
        };

        Ok(Self { hashed_address, info })
    }
}

/// Trie changeset entry representing the state of a trie node before a block.
///
/// `nibbles` is the subkey when used as a value in the changeset tables.
/// This is a local definition since the upstream `reth-trie-common` crate does
/// not provide this type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrieChangeSetsEntry {
    /// The nibbles of the intermediate node
    pub nibbles: StoredNibblesSubKey,
    /// Node value prior to the block being processed, None indicating it didn't exist.
    pub node: Option<BranchNodeCompact>,
}

impl ValueWithSubKey for TrieChangeSetsEntry {
    type SubKey = StoredNibblesSubKey;

    fn get_subkey(&self) -> Self::SubKey {
        self.nibbles.clone()
    }
}

impl Compress for TrieChangeSetsEntry {
    type Compressed = Vec<u8>;

    fn compress_to_buf<B: BufMut + AsMut<[u8]>>(&self, buf: &mut B) {
        let _ = self.nibbles.to_compact(buf);
        if let Some(ref node) = self.node {
            let _ = node.to_compact(buf);
        }
    }
}

impl Decompress for TrieChangeSetsEntry {
    fn decompress(value: &[u8]) -> Result<Self, DatabaseError> {
        if value.is_empty() {
            return Ok(Self {
                nibbles: StoredNibblesSubKey::from(reth_trie_common::Nibbles::default()),
                node: None,
            });
        }

        let (nibbles, rest) = StoredNibblesSubKey::from_compact(value, 65);
        let node = if rest.is_empty() { None } else { Some(BranchNodeCompact::from_compact(rest, rest.len()).0) };
        Ok(Self { nibbles, node })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reth_db::table::{Compress, Decompress};

    #[test]
    fn test_hashed_account_before_tx_roundtrip_some() {
        let original = HashedAccountBeforeTx {
            hashed_address: B256::repeat_byte(0xaa),
            info: Some(Account {
                nonce: 42,
                balance: alloy_primitives::U256::from(1000u64),
                bytecode_hash: None,
            }),
        };

        let compressed = original.clone().compress();
        assert!(compressed.len() > 32, "Should contain address + account data");

        let decompressed = HashedAccountBeforeTx::decompress(&compressed).unwrap();
        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_hashed_account_before_tx_roundtrip_none() {
        let original = HashedAccountBeforeTx {
            hashed_address: B256::repeat_byte(0xbb),
            info: None,
        };

        let compressed = original.clone().compress();
        assert_eq!(compressed.len(), 32, "None account should be just the 32-byte address");

        let decompressed = HashedAccountBeforeTx::decompress(&compressed).unwrap();
        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_hashed_account_before_tx_subkey() {
        let addr = B256::repeat_byte(0xcc);
        let entry = HashedAccountBeforeTx::new(addr, None);
        assert_eq!(entry.get_subkey(), addr);
    }
}
