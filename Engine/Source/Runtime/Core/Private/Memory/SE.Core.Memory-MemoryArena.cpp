module;
#include <cstdint>
#include <list>
module SE.Core.Memory:MemoryArena;
import SE.Core.Memory;

namespace SIByL::Core
{
	MemoryArena::~MemoryArena() {
		FreeAligned(currentBlock);
		for (auto& block : usedBlocks) FreeAligned(block.second);
		for (auto& block : availableBlocks) FreeAligned(block.second);
	}

	auto MemoryArena::alloc(size_t nBytes) noexcept -> void*
	{
		// round up nBytes to minimum machine alignment
		nBytes = ((nBytes + 15) & (~15));
		if (currentAllocSize + nBytes > currentAllocSize) {
			// add current block to usedBlocks list
			if (currentBlock) {
				usedBlocks.push_back(std::make_pair(currentAllocSize, currentBlock));
				currentBlock = nullptr;
			}
			// get new block of memory for memory arena
			// - Try to get memory block from availableBlocks
			for (auto iter = availableBlocks.begin(); iter != availableBlocks.end(); iter++) {
				if (iter->first >= nBytes) {
					currentAllocSize = iter->first;
					currentBlock = iter->second;
					availableBlocks.erase(iter);
					break;
				}
			}
			if (!currentBlock) {
				currentAllocSize = std::max(nBytes, blockSize);
				currentBlock = AllocAligned<uint8_t>(currentAllocSize);
			}
			currentBlockPos = 0;
		}
		void* ret = currentBlock + currentBlockPos;
		currentBlockPos += nBytes;
		return ret;
	}

	auto MemoryArena::reset() noexcept -> void {
		currentBlockPos = 0;
		availableBlocks.splice(availableBlocks.begin(), usedBlocks);
	}

	auto MemoryArena::totalAllocated() const noexcept -> size_t {
		size_t total = currentAllocSize;
		for (auto const& alloc : usedBlocks) total += alloc.first;
		for (auto const& alloc : availableBlocks) total += alloc.first;
		return total;
	}

}