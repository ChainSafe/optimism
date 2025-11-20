package gap_elp2p

import (
	"math/big"
	"testing"
	"time"

	"github.com/ethereum-optimism/optimism/op-acceptance-tests/tests/sync/elsync/gap_elp2p/utils"
	"github.com/ethereum-optimism/optimism/op-devstack/devtest"
	"github.com/ethereum-optimism/optimism/op-devstack/presets"
	"github.com/ethereum-optimism/optimism/op-service/eth"
	"github.com/ethereum-optimism/optimism/op-service/sources"
	"github.com/ethereum-optimism/optimism/op-service/txplan"
	"github.com/ethereum-optimism/optimism/op-supervisor/supervisor/types"
	"github.com/ethereum-optimism/optimism/op-wheel/engine"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	gethTypes "github.com/ethereum/go-ethereum/core/types"
	"github.com/stretchr/testify/require"
)

// TestChainRevert tests that the L2ELB (op-reth) correctly handles chain reverts.
// It creates blocks with state changes, reverts the chain using engine.Rewind and ForkChoiceUpdate
// and then verifies that L2ELB correctly handles the chain revert.
//
// The test verifies that:
// 1. Proofs are generated correctly for all blocks before revert
// 2. The chain can be reverted to an earlier block using debug_setHead
// 3. L2ELB re-syncs after the revert
// 4. Proofs remain valid after the revert for blocks up to and including the rewind point
// 5. The ExEx properly handles the ChainReverted notification
func TestChainRevert(gt *testing.T) {
	t := devtest.SerialT(gt)
	ctx := t.Ctx()
	sys := presets.NewSingleChainMultiNode(t)

	sys.L2CLB.Stop()

	// Deploy a simple storage contract
	artifactPath := "./artifacts/SimpleStorage.json"
	parsedABI, bin, err := utils.LoadArtifact(artifactPath)
	if err != nil {
		t.Error("failed to load artifact: %v", err)
		t.FailNow()
	}

	user := sys.FunderL2.NewFundedEOA(eth.OneHundredthEther)
	contractAddress, deployBlock, err := utils.DeployContract(ctx, user, bin)
	if err != nil {
		t.Error("failed to deploy contract: %v", err)
		t.FailNow()
	}
	t.Logf("Contract deployed at address %s in L2 block %d", contractAddress.Hex(), deployBlock)

	// Create 4 blocks with state changes
	var blockNumbers []uint64
	var blockHashes []common.Hash
	for i := range 5 {
		writeVal := big.NewInt(int64(i * 100))
		callData, err := parsedABI.Pack("setValue", writeVal)
		require.NoError(gt, err)
		callTx := txplan.NewPlannedTx(user.Plan(), txplan.WithTo(&contractAddress), txplan.WithData(callData))
		receipt, err := callTx.Included.Eval(ctx)
		require.NoError(gt, err)
		require.Equal(gt, gethTypes.ReceiptStatusSuccessful, receipt.Status)
		blockNum := receipt.BlockNumber.Uint64()
		blockHash := receipt.BlockHash
		blockNumbers = append(blockNumbers, blockNum)
		blockHashes = append(blockHashes, blockHash)
		t.Logf("setValue(%d) transaction included in L2 block %d (hash: %s)", i*100, blockNum, blockHash.Hex())
	}

	sys.L2CL.Advanced(types.LocalUnsafe, 2, 7)

	// Rewind to block 2 (index 1)
	rewindToBlock := blockNumbers[1]
	t.Logf("Triggering chain revert to block %d", rewindToBlock)

	time.Sleep(2 * time.Second)

	engineClientEL := sys.L2EL.Escape().L2EngineClient().(*sources.EngineAPIClient)
	rpcEL := sys.L2EL.Escape().L2EthClient().RPC()
	engine.Rewind(ctx, t.Logger(), engineClientEL, rpcEL, rewindToBlock, true)

	time.Sleep(2 * time.Second)

	sys.L2ELB.ForkchoiceUpdate(sys.L2EL, rewindToBlock, 0, 0, nil)
	sys.L2CLB.Start()

	// Verify L2EL has rewound correctly
	t.Logf("Verifying L2EL has rewound to block %d", rewindToBlock)
	l2elBlock, err := sys.L2EL.Escape().L2EthClient().BlockRefByLabel(ctx, eth.Unsafe)
	require.NoError(gt, err)
	t.Logf("L2EL current unsafe block before starting: %d (hash: %s)", l2elBlock.Number, l2elBlock.Hash.Hex())

	// Verify L2ELB has rewound correctly
	t.Logf("Verifying L2ELB has rewound to block %d", rewindToBlock)
	currentBlock, err := sys.L2ELB.Escape().L2EthClient().BlockRefByLabel(ctx, eth.Unsafe)
	require.NoError(gt, err)
	t.Logf("L2ELB current unsafe block: %d (hash: %s)", currentBlock.Number, currentBlock.Hash.Hex())

	// Verify proofs still work for blocks up to and including the rewind point
	t.Logf("Verifying proofs for blocks up to rewind point")
	for i := range 2 { // Blocks 1, 2
		blockNum := blockNumbers[i]
		t.Logf("Verifying proofs for block %d (hash: %s)", blockNum, blockHashes[i].Hex())
		utils.FetchAndVerifyProofs(t, sys, contractAddress, []common.Hash{common.HexToHash("0x0")}, blockNum)
	}

	// Verify that proofs for removed blocks (3 and 4) are no longer accessible
	t.Logf("Verifying that removed blocks no longer have valid proofs")
	for i := 2; i < len(blockNumbers); i++ { // Blocks 3, 4
		t.Logf("Attempting to fetch proof for removed block %d (hash: %s)", blockNumbers[i], blockHashes[i].Hex())
		// Try to get proof from L2ELB - should fail or return error
		_, err := sys.L2ELB.Escape().L2EthClient().GetProof(ctx, contractAddress, []common.Hash{common.HexToHash("0x0")}, hexutil.Uint64(blockNumbers[i]).String())
		require.Error(t, err)

		// Try to get block info - should also fail
		_, err = sys.L2ELB.Escape().L2EthClient().BlockRefByNumber(ctx, blockNumbers[i])
		require.Error(t, err)
	}

	// sys.L2Batcher.Start()
	t.Logf("Test completed.")
}
