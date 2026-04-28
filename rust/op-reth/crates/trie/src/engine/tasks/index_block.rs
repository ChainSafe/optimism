use super::super::state::EngineState;
use crate::{engine::EngineError, BlockStateDiff, OpProofsStore};
use alloy_eips::eip1898::BlockWithParent;
use crossbeam_channel::Sender;
use reth_evm::ConfigureEvm;
use reth_primitives_traits::BlockTy;
use reth_provider::{
    BlockHashReader, BlockReader, DatabaseProviderFactory, StateProviderFactory, StateReader,
};
use reth_trie_common::{updates::TrieUpdatesSorted, HashedPostStateSorted};
use std::time::Instant;
use tracing::{debug, info};

pub(crate) struct IndexBlockTask {
    pub(crate) block: BlockWithParent,
    pub(crate) sorted_trie_updates: TrieUpdatesSorted,
    pub(crate) sorted_post_state: HashedPostStateSorted,
    pub(crate) reply: Sender<Result<(), EngineError>>,
}

impl IndexBlockTask {
    pub(crate) fn execute<Evm, Provider, Store>(
        self,
        state: &mut EngineState<Evm, Provider, Store>,
    ) where
        Evm: ConfigureEvm,
        Provider: BlockHashReader
            + StateReader
            + DatabaseProviderFactory
            + StateProviderFactory
            + BlockReader<Block = BlockTy<Evm::Primitives>>
            + Clone
            + 'static,
        Store: OpProofsStore + Clone + 'static,
    {
        let IndexBlockTask { block, sorted_trie_updates, sorted_post_state, reply } = self;
        let result = run(state, block, sorted_trie_updates, sorted_post_state);
        let _ = reply.send(result);
    }
}

fn run<Evm, Provider, Store>(
    state: &mut EngineState<Evm, Provider, Store>,
    block: BlockWithParent,
    sorted_trie_updates: TrieUpdatesSorted,
    sorted_post_state: HashedPostStateSorted,
) -> Result<(), EngineError>
where
    Evm: ConfigureEvm,
    Provider: BlockHashReader
        + StateReader
        + DatabaseProviderFactory
        + StateProviderFactory
        + BlockReader<Block = BlockTy<Evm::Primitives>>
        + Clone
        + 'static,
    Store: OpProofsStore + Clone + 'static,
{
    let start = Instant::now();
    let tip = state.get_tip()?;

    if block.block.number <= tip.number {
        debug!(
            block_number = block.block.number,
            tip_number = tip.number,
            "Block already covered by tip, skipping store_block_updates",
        );
        return Ok(());
    }

    if block.block.number > tip.number.saturating_add(1) {
        debug!(
            block_number = block.block.number,
            tip_number = tip.number,
            "Gap detected, updating sync target",
        );
        if block.block.number > state.sync_target {
            state.sync_target = block.block.number;
        }
        return Ok(());
    }

    if block.parent != tip.hash {
        return Err(EngineError::ParentHashMismatch {
            block_number: block.block.number,
            expected_parent_hash: tip.hash,
            actual_parent_hash: block.parent,
        });
    }

    state.memory.insert(block, BlockStateDiff { sorted_trie_updates, sorted_post_state });

    let total_duration = start.elapsed();

    #[cfg(feature = "metrics")]
    state.metrics.total_duration_seconds.record(total_duration);

    info!(
        block_number = block.block.number,
        ?total_duration,
        "Trie updates buffered successfully",
    );

    Ok(())
}
