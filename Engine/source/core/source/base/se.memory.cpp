#define DLIB_EXPORT
#include <se.core.hpp>
#undef DLIB_EXPORT
#include <format>
#include <fstream>

namespace se {
#ifndef ALIGN
#define ALIGN(x, a) (((x) + ((a)-1)) & ~((a)-1))
#endif

struct BlockHeader {
  // union-ed with data
  BlockHeader* pNext;
};

struct PageHeader {
  // followed by blocks in this page
  PageHeader* pNext;
  // helper function that gives the first block
  auto blocks() noexcept -> BlockHeader* {
    return reinterpret_cast<BlockHeader*>(this + 1);
  }
};

struct Allocator {
  Allocator(); ~Allocator();
  Allocator(size_t data_size, size_t page_size, size_t alignment);
  // resets the allocator to a new configuration
  auto reset(size_t data_size, size_t page_size, size_t alignment) noexcept
      -> void;
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
  size_t dataSize;
  size_t pageSize;
  size_t alignmentSize;
  size_t blockSize;
  uint32_t blocksPerPage;
  // statistics
  uint32_t numPages;
  uint32_t numBlocks;
  uint32_t numFreeBlocks;
  // gets the next block
  auto nextBlock(BlockHeader* pBlock) noexcept -> BlockHeader*;
  // disable copy & assignment
  Allocator(const Allocator& clone) = delete;
  auto operator=(const Allocator& rhs) -> Allocator& = delete;
};


Allocator::Allocator() : pPageList(nullptr), pFreeList(nullptr) {}

Allocator::Allocator(size_t data_size, size_t page_size, size_t alignment)
    : pPageList(nullptr), pFreeList(nullptr) {
  reset(data_size, page_size, alignment);
}

Allocator::~Allocator() { freeAll(); }

auto Allocator::reset(size_t data_size, size_t page_size,
                      size_t alignment) noexcept -> void {
  freeAll();

  dataSize = data_size;
  pageSize = page_size;

  size_t minimal_size =
      (sizeof(BlockHeader) > dataSize) ? sizeof(BlockHeader) : dataSize;
  // this magic only works when alignment is 2^n, which should general be the
  // case because most CPU/GPU also requires the aligment be in 2^n but still we
  // use a assert to guarantee it
  blockSize = ALIGN(minimal_size, alignment);

  alignmentSize = blockSize - minimal_size;

  blocksPerPage = uint32_t((pageSize - sizeof(PageHeader)) / blockSize);
}

auto Allocator::allocate() noexcept -> void* {
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

auto Allocator::free(void* p) noexcept -> void {
  BlockHeader* block = reinterpret_cast<BlockHeader*>(p);
  block->pNext = pFreeList;
  pFreeList = block;
  ++numFreeBlocks;
}

auto Allocator::freeAll() noexcept -> void {
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

auto Allocator::nextBlock(BlockHeader* pBlock) noexcept -> BlockHeader* {
  return reinterpret_cast<BlockHeader*>(reinterpret_cast<uint8_t*>(pBlock) +
                                        blockSize);
}

// size_t* Memory::pBlockSizeLookup;
// Allocator* Memory::pAllocators;

static const uint32_t kBlockSizes[] = {
  // 4-increments
  4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76,
  80, 84, 88, 92, 96,
  // 32-increments
  128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576,
  608, 640,
  // 64-increments
  704, 768, 832, 896, 960, 1024 };

static const uint32_t kPageSize = 8192;
static const uint32_t kAlignment = 4;

// number of elements in the block size array
static const uint32_t kNumBlockSizes =
sizeof(kBlockSizes) / sizeof(kBlockSizes[0]);

// largest valid block size
static const uint32_t kMaxBlockSize = kBlockSizes[kNumBlockSizes - 1];

struct MemoryManager {
  SINGLETON(MemoryManager, {
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
  });
  ~MemoryManager();
  auto allocate(size_t size) noexcept -> void*;
  auto free(void* p, size_t size) noexcept -> void;
 private:
  size_t* pBlockSizeLookup;
  Allocator* pAllocators;
  auto lookUpAllocator(size_t size) noexcept -> Allocator*;
};

MemoryManager::~MemoryManager() {

}

#define L1_CACHE_LINE_SIZE 64

inline auto AllocAligned(size_t size) -> void* {
  return _aligned_malloc(size, L1_CACHE_LINE_SIZE);
}

template <class T>
inline auto AllocAligned(size_t count) -> T* {
  return (T*)AllocAligned(count * sizeof(T));
}

inline auto FreeAligned(void* p) -> void { _aligned_free(p); }

auto MemoryManager::allocate(size_t size) noexcept -> void* {
  Allocator* pAlloc = lookUpAllocator(size);
  if (pAlloc) return pAlloc->allocate();
  else return AllocAligned(size);
}

auto MemoryManager::free(void* p, size_t size) noexcept -> void {
  Allocator* pAlloc = lookUpAllocator(size);
  if (pAlloc) pAlloc->free(p);
  else FreeAligned(p);
}

auto MemoryManager::lookUpAllocator(size_t size) noexcept -> Allocator* {
  // check eligibility for lookup
  if (size <= kMaxBlockSize) return pAllocators + pBlockSizeLookup[size];
  else return nullptr;
}

auto root::memory::allocate(size_t size) noexcept -> void* {
  return Singleton<MemoryManager>::instance()->allocate(size);
}

auto root::memory::free(void* p, size_t size) noexcept -> void {
  Singleton<MemoryManager>::instance()->free(p, size);
}

buffer::buffer() : data(nullptr), size(0), isReference(false) {}
buffer::buffer(size_t size) : size(size), isReference(false) 
  { data = root::memory::allocate(size); }
buffer::buffer(buffer const& b) {
  release(); size = b.size; data = root::memory::allocate(size);
  isReference = false; memcpy(data, b.data, size); }
buffer::buffer(buffer&& b) {
  release(); size = b.size; data = b.data;
  isReference = false; b.data = nullptr; b.size = 0; }
buffer::buffer(void* data, size_t size)
  : data(data), size(size), isReference(true) {}
buffer::~buffer() { release(); }
auto buffer::operator=(buffer const& b) -> buffer& {
  release(); size = b.size; data = root::memory::allocate(size);
  memcpy(data, b.data, size); return *this; }
auto buffer::operator=(buffer&& b) -> buffer& {
  release(); size = b.size; data = b.data;
  b.data = nullptr; b.size = 0; return *this; }
auto buffer::release() noexcept -> void {
  if (data == nullptr || isReference) return;
  root::memory::free(data, size);
  data = nullptr; size = 0;
}

auto syncReadFile(char const* path, se::buffer& buffer) noexcept -> bool {
  std::ifstream ifs(path, std::ifstream::binary);
  if (ifs.is_open()) {
    ifs.seekg(0, std::ios::end);
    size_t size = size_t(ifs.tellg());
    buffer = se::buffer(size + 1);
    buffer.size = size;
    ifs.seekg(0);
    ifs.read(reinterpret_cast<char*>(buffer.data), size);
    ((char*)buffer.data)[size] = '\0';
    ifs.close();
  } else {
    root::print::error(std::format(
    "Core.IO:SyncRW::syncReadFile() failed, file \'{}\' not found.", path));
  }
  return false;
}

auto syncWriteFile(char const* path, se::buffer& buffer) noexcept -> bool {
  std::ofstream ofs(path, std::ios::out | std::ios::binary);
  if (ofs.is_open()) {
    ofs.write((char*)buffer.data, buffer.size);
    ofs.close();
  } else {
    root::print::error(std::format(
    "Core.IO:SyncRW::syncWriteFile() failed, file \'{}\' open failed.", path));
  }
  return false;
}

auto syncWriteFile(char const* path, std::vector<se::buffer*> const& buffers) noexcept -> bool {
  std::ofstream ofs(path, std::ios::out | std::ios::binary);
  if (ofs.is_open()) {
    for (auto* buffer : buffers) ofs.write((char*)buffer->data, buffer->size);
    ofs.close();
  } else {
    root::print::error(std::format(
    "Core.IO:SyncRW::syncWriteFile() failed, file \'{}\' open failed.", path));
  }
  return false;
}
}