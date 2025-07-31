package helpers

import (
	"github.com/ethereum-optimism/optimism/cannon/mipsevm/arch"
	"github.com/ethereum-optimism/optimism/cannon/mipsevm/memory"
	"github.com/ethereum-optimism/optimism/cannon/mipsevm/multithreaded"
	"github.com/ethereum-optimism/optimism/cannon/mipsevm/testutil"
)

func StoreInstructionWithCacheUpdate(mem *memory.Memory, pc arch.Word, insn uint32, vm any) {
	testutil.StoreInstruction(mem, pc, insn)
	if instVm, ok := vm.(*multithreaded.InstrumentedState); ok {
		instVm.UpdateInstructionCache(pc)
	}
}
