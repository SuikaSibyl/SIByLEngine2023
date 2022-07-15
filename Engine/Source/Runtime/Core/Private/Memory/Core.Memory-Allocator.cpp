module;
#include <cstdint>
module Core.Memory:Allocator;
import Core.Memory;

#ifndef ALIGN
#define ALIGN(x, a) (((x) + ((a) - 1)) & ~((a) - 1))
#endif

namespace SIByL::Core
{
	auto PageHeader::blocks() noexcept -> BlockHeader* {
		return reinterpret_cast<BlockHeader*>(this + 1);
	}

	Allocator::Allocator()
		:pPageList(nullptr), pFreeList(nullptr)
	{}

	Allocator::Allocator(size_t data_size, size_t page_size, size_t alignment)
		: pPageList(nullptr), pFreeList(nullptr)
	{
		reset(data_size, page_size, alignment);
	}

	Allocator::~Allocator()
	{
		freeAll();
	}

	auto Allocator::reset(size_t data_size, size_t page_size, size_t alignment) noexcept -> void
	{
		freeAll();

		dataSize = data_size;
		pageSize = page_size;

		size_t minimal_size = (sizeof(BlockHeader) > dataSize) ? sizeof(BlockHeader) : dataSize;
		// this magic only works when alignment is 2^n, which should general be the case
		// because most CPU/GPU also requires the aligment be in 2^n
		// but still we use a assert to guarantee it
		blockSize = ALIGN(minimal_size, alignment);

		alignmentSize = blockSize - minimal_size;

		blocksPerPage = (pageSize - sizeof(PageHeader)) / blockSize;
	}

	auto Allocator::allocate() noexcept -> void*
	{
		if (!pFreeList) {
			// allocate a new page
			PageHeader* pNewPage = reinterpret_cast<PageHeader*>(new uint8_t[pageSize]);
			pNewPage->pNext = nullptr;

			++numPages;
			numBlocks += blocksPerPage;
			numFreeBlocks += blocksPerPage;

			if (pPageList) {
				pNewPage->pNext = pPageList;
			}

			pPageList = pNewPage;

			BlockHeader* pBlock = pNewPage->blocks();
			// link each block in the page
			for (uint32_t i = 0; i < blocksPerPage - 1; i++) {
				pBlock->pNext = nextBlock(pBlock);
				pBlock = nextBlock(pBlock);
			}
			pBlock->pNext = nullptr;

			pFreeList = pNewPage->blocks();
		}

		BlockHeader* freeBlock = pFreeList;
		pFreeList = pFreeList->pNext;
		--numFreeBlocks;

		return reinterpret_cast<void*>(freeBlock);
	}

	auto Allocator::free(void* p) noexcept -> void
	{
		BlockHeader* block = reinterpret_cast<BlockHeader*>(p);
		block->pNext = pFreeList;
		pFreeList = block;
		++numFreeBlocks;
	}

	auto Allocator::freeAll() noexcept -> void
	{
		PageHeader* pPage = pPageList;
		while (pPage) {
			PageHeader* _p = pPage;
			pPage = pPage->pNext;

			delete[] reinterpret_cast<uint8_t*>(_p);
		}

		pPageList = nullptr;
		pFreeList = nullptr;

		numPages = 0;
		numBlocks = 0;
		numFreeBlocks = 0;
	}

	auto Allocator::nextBlock(BlockHeader* pBlock) noexcept -> BlockHeader*
	{
		return reinterpret_cast<BlockHeader*>(reinterpret_cast<uint8_t*>(pBlock) + blockSize);
	}

}