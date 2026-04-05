//! State management for the live trie collector.

use crate::{
    overlay_provider::MemoryOverlayOpProofsStateProviderRef, provider::OpProofsStateProviderRef,
    BlockStateDiff, OpProofsProviderRO,
};
use alloy_eips::eip1898::BlockWithParent;
use alloy_primitives::{map::HashMap, B256};
use parking_lot::RwLock;
use std::{collections::BTreeMap, sync::Arc};

/// Buffer for holding blocks waiting to be persisted.
///
/// This acts as the in-memory "tip" of the chain for the trie calculator.
#[derive(Debug, Default)]
pub(crate) struct InMemoryState {
    /// All blocks that are not on disk yet.
    pub(crate) blocks: RwLock<HashMap<B256, Arc<(BlockWithParent, BlockStateDiff)>>>,
    /// Mapping of block numbers to block hashes.
    pub(crate) numbers: RwLock<BTreeMap<u64, B256>>,
}

impl InMemoryState {
    /// Create a new empty in-memory state.
    pub(crate) fn new() -> Self {
        Self {
            blocks: RwLock::new(HashMap::default()),
            numbers: RwLock::new(BTreeMap::new()),
        }
    }

    /// Insert a block into the buffer.
    pub(crate) fn insert(&self, block: BlockWithParent, diff: BlockStateDiff) {
        let hash = block.block.hash;
        let number = block.block.number;
        let state = Arc::new((block, diff));

        // Write locks
        let mut blocks = self.blocks.write();
        let mut numbers = self.numbers.write();

        blocks.insert(hash, state);
        numbers.insert(number, hash);
    }

    /// Returns the number of buffered blocks.
    pub(crate) fn len(&self) -> usize {
        self.blocks.read().len()
    }

    /// Returns true if the buffer is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.blocks.read().is_empty()
    }

    /// Clear the buffer.
    pub(crate) fn clear(&self) {
        let mut blocks = self.blocks.write();
        let mut numbers = self.numbers.write();
        blocks.clear();
        numbers.clear();
    }

    /// Prunes blocks from the buffer that are strictly before the given block number.
    pub(crate) fn prune_before(&self, number: u64) {
        let mut blocks = self.blocks.write();
        let mut numbers = self.numbers.write();

        // Identify block numbers to remove
        let mut to_remove = Vec::new();
        // Use BTreeMap's ordered nature
        for (&num, &hash) in numbers.iter() {
            if num < number {
                to_remove.push((num, hash));
            } else {
                break;
            }
        }

        for (num, hash) in to_remove {
            numbers.remove(&num);
            blocks.remove(&hash);
        }
    }

    /// Removes blocks starting from `from` (inclusive) through the tip.
    ///
    /// Mirrors the disk `unwind_history(to)` semantics where `to.block.number` is the
    /// first block removed. After this call, only blocks with number < `from` remain.
    pub(crate) fn unwind(&self, from: u64) {
        let mut blocks = self.blocks.write();
        let mut numbers = self.numbers.write();

        let mut to_remove = Vec::new();
        for (&num, &hash) in numbers.iter().rev() {
            if num >= from {
                to_remove.push((num, hash));
            } else {
                break;
            }
        }

        for (num, hash) in to_remove {
            numbers.remove(&num);
            blocks.remove(&hash);
        }
    }

    /// Returns the state for a given block hash.
    pub(crate) fn state_by_hash(&self, hash: B256) -> Option<Arc<(BlockWithParent, BlockStateDiff)>> {
        self.blocks.read().get(&hash).cloned()
    }

    /// Returns the hash for a specific block number
    pub(crate) fn hash_by_number(&self, number: u64) -> Option<B256> {
        self.numbers.read().get(&number).copied()
    }

    /// Returns the state for a given block number.
    pub(crate) fn state_by_number(&self, number: u64) -> Option<Arc<(BlockWithParent, BlockStateDiff)>> {
        let hash = self.hash_by_number(number)?;
        self.state_by_hash(hash)
    }
}

/// Manager for the in-memory state of the live trie.
#[derive(Debug, Clone, Default)]
pub struct LiveTrieState {
    inner: Arc<InMemoryState>,
}

impl LiveTrieState {
    /// Create a new live trie state manager.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(InMemoryState::new()),
        }
    }

    /// Insert a block into the buffer.
    pub fn insert(&self, block: BlockWithParent, diff: BlockStateDiff) {
        self.inner.insert(block, diff);
    }

    /// Returns the number of buffered blocks.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Clear the buffer.
    pub fn clear(&self) {
        self.inner.clear();
    }

    /// Return a reference to the inner in-memory state.
    pub(crate) fn inner(&self) -> Arc<InMemoryState> {
        self.inner.clone()
    }

    /// Prunes blocks from the buffer that are strictly before the given block number.
    pub fn prune_before(&self, number: u64) {
        self.inner.prune_before(number);
    }

    /// Removes blocks starting from `from` (inclusive) through the tip.
    pub fn unwind(&self, from: u64) {
        self.inner.unwind(from);
    }

    /// Returns the state for a given block hash.
    pub fn state_by_hash(&self, hash: B256) -> Option<Arc<(BlockWithParent, BlockStateDiff)>> {
        self.inner.state_by_hash(hash)
    }

    /// Returns the state for a given block number.
    pub fn state_by_number(&self, number: u64) -> Option<Arc<(BlockWithParent, BlockStateDiff)>> {
        self.inner.state_by_number(number)
    }

    /// Return state provider with reference to in-memory blocks that overlay storage state.
    ///
    /// This retrieves the chain of blocks ending at `hash` from the in-memory buffer,
    /// providing a view that includes both the buffered changes and the underlying disk state.
    pub fn state_provider<'a, P>(
        &self,
        hash: B256,
        inner: OpProofsStateProviderRef<'a, P>,
    ) -> MemoryOverlayOpProofsStateProviderRef<'a, P>
    where
        P: OpProofsProviderRO + Clone,
    {
        let mut in_memory = Vec::new();
        let blocks = self.inner.blocks.read();

        // Trace back from the requested hash to finding no parent in memory
        let mut current_hash = hash;
        while let Some(state) = blocks.get(&current_hash) {
            in_memory.push(state.clone());
            current_hash = state.0.parent;
        }

        // The vector is currently Newest -> Oldest. Reverse it to Oldest -> Newest
        // as expected by the overlay provider for correct replay.
        in_memory.reverse();

        MemoryOverlayOpProofsStateProviderRef::new(inner, in_memory)
    }
}
