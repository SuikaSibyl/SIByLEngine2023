module;
#include <cstdint>
#include <list>
export module Core.Memory:MemoryArena;

namespace SIByL::Core
{
	/**
	* Two advantages:
	* 1. allocation is extremely fast, usually just a pointer increment
	* 2. improve locality of reference and lead to fewer cache misses
	*/
	export struct MemoryArena
	{
		/* Allocate in memory chunks in size of blockSize, which is defaultly 256kB */
		MemoryArena(size_t blockSize = 262144) :blockSize(blockSize) {}
		/** Only free all of the memory in the arena at once */
		~MemoryArena();

		/** allocation request */
		auto alloc(size_t nBytes) noexcept -> void*;

		/** allocation request to allocate an array of objects of the given type.*/
		template <class T>
		auto alloc(size_t n = 1, bool runConstructor = true) noexcept -> T*;

		auto reset() noexcept -> void;
		auto totalAllocated() const noexcept -> size_t;

	private:
		size_t currentBlockPos = 0, currentAllocSize = 0;
		uint8_t* currentBlock = nullptr;
		size_t const blockSize;
		std::list<std::pair<size_t, uint8_t*>> usedBlocks, availableBlocks;

	private:
		MemoryArena(MemoryArena const&) = delete;
		auto operator=(MemoryArena const&) -> MemoryArena& = delete;
	};

	template <class T>
	auto MemoryArena::alloc(size_t n, bool runConstructor) noexcept -> T* {
		T* ret = (T*)alloc(n * sizeof(T));
		if (runConstructor)
			for (size_t i = 0; i < n; i++)
				new(&ret[i])T();
		return ret;
	}
}