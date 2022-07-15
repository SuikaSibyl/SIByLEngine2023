module;
#include <cstdint>
#include <utility>
#include <type_traits>
module Core.Memory:MemoryManager;
import Core.Memory;
import :Allocator;

namespace SIByL::Core
{
	//size_t* Memory::pBlockSizeLookup;
	//Allocator* Memory::pAllocators;

	static const uint32_t kBlockSizes[] = {
		// 4-increments
		4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48,
		52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96,

		// 32-increments
		128, 160, 192, 224, 256, 288, 320, 352, 384,
		416, 448, 480, 512, 544, 576, 608, 640,

		// 64-increments
		704, 768, 832, 896, 960, 1024
	};

	static const uint32_t kPageSize = 8192;
	static const uint32_t kAlignment = 4;

	// number of elements in the block size array
	static const uint32_t kNumBlockSizes =
		sizeof(kBlockSizes) / sizeof(kBlockSizes[0]);

	// largest valid block size
	static const uint32_t kMaxBlockSize =
		kBlockSizes[kNumBlockSizes - 1];

	auto MemoryManager::startUp() noexcept -> void
	{
		// initialize block size lookup table
		pBlockSizeLookup = new size_t[kMaxBlockSize + 1];
		size_t j = 0;
		for (size_t i = 0; i <= kMaxBlockSize; i++) {
			if (i > kBlockSizes[j]) ++j;
			pBlockSizeLookup[i] = j;
		}

		// initialize the allocators
		pAllocators = new Allocator[kNumBlockSizes];
		for (size_t i = 0; i < kNumBlockSizes; i++) {
			pAllocators[i].reset(kBlockSizes[i], kPageSize, kAlignment);
		}

		// duplicate initialize
		if (singleton != nullptr) __debugbreak();
		singleton = this;
	}

	auto MemoryManager::shutDown() noexcept -> void
	{
		if (pAllocators) delete[] pAllocators;
		if (pBlockSizeLookup) delete[] pBlockSizeLookup;

		singleton = nullptr;
	}

	auto MemoryManager::allocate(size_t size) noexcept -> void*
	{
		Allocator* pAlloc = lookUpAllocator(size);
		if (pAlloc)
			return pAlloc->allocate();
		else
			return ::malloc(size);
	}

	auto MemoryManager::free(void* p, size_t size) noexcept -> void
	{
		Allocator* pAlloc = lookUpAllocator(size);
		if (pAlloc)
			pAlloc->free(p);
		else
			::free(p);
	}

	auto MemoryManager::lookUpAllocator(size_t size) noexcept -> Allocator*
	{
		// check eligibility for lookup
		if (size <= kMaxBlockSize)
			return pAllocators + pBlockSizeLookup[size];
		else
			return nullptr;
	}

}