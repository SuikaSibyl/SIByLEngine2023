module;
#include <cstdint>
export module SE.Core.Memory:Allocator;

namespace SIByL::Core
{
    struct BlockHeader {
        // union-ed with data
        BlockHeader* pNext;
    };

    struct PageHeader {
        // followed by blocks in this page
        PageHeader* pNext;
        // helper function that gives the first block
        auto blocks() noexcept -> BlockHeader*;
    };

	export struct Allocator
	{
        Allocator();
        Allocator(size_t data_size, size_t page_size, size_t alignment);
        ~Allocator();

        // resets the allocator to a new configuration
        auto reset(size_t data_size, size_t page_size, size_t alignment) noexcept -> void;
        // alloc and free blocks
        auto allocate() noexcept -> void*;
        auto free(void* p) noexcept -> void;
        auto freeAll() noexcept -> void;

    private:
        // the page list
        PageHeader* pPageList = nullptr;
        // the free block list
        BlockHeader* pFreeList;

        // size definition
        size_t      dataSize;
        size_t      pageSize;
        size_t      alignmentSize;
        size_t      blockSize;
        uint32_t    blocksPerPage;

        // statistics
        uint32_t    numPages;
        uint32_t    numBlocks;
        uint32_t    numFreeBlocks;

        // gets the next block
        auto nextBlock(BlockHeader* pBlock) noexcept -> BlockHeader*;
        // disable copy & assignment
        Allocator(const Allocator& clone) = delete;
        auto operator=(const Allocator& rhs) ->Allocator& = delete;
	};
}