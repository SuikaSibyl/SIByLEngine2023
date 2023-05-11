#pragma once
#include "SE.Core.Memory.hpp"

#include <cstdint>
#include <list>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace SIByL::Core {
#ifndef ALIGN
#define ALIGN(x, a) (((x) + ((a)-1)) & ~((a)-1))
#endif

auto PageHeader::blocks() noexcept -> BlockHeader* {
  return reinterpret_cast<BlockHeader*>(this + 1);
}

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

Buffer::Buffer() : data(nullptr), size(0) {}

Buffer::Buffer(size_t size) : size(size) { data = Alloc(size); }

Buffer::Buffer(Buffer const& b) {
  release();
  size = b.size;
  data = Alloc(size);
  memcpy(data, b.data, size);
}

Buffer::Buffer(Buffer&& b) {
  release();
  size = b.size;
  data = b.data;
  b.data = nullptr;
  b.size = 0;
}

Buffer::~Buffer() { release(); }

auto Buffer::operator=(Buffer const& b) -> Buffer& {
  release();
  size = b.size;
  data = Alloc(size);
  memcpy(data, b.data, size);
  return *this;
}

auto Buffer::operator=(Buffer&& b) -> Buffer& {
  release();
  size = b.size;
  data = b.data;
  b.data = nullptr;
  b.size = 0;
  return *this;
}

auto Buffer::release() noexcept -> void {
  if (data == nullptr) return;
  Free(data, size);
  data = nullptr;
  size = 0;
}

auto Buffer::stream() noexcept -> BufferStream {
  return BufferStream{reinterpret_cast<char*>(data)};
}

auto BufferStream::operator<<(char c) -> BufferStream& {
  data[0] = c;
  data++;
  return *this;
}

auto BufferStream::operator<<(std::string const& str) -> BufferStream& {
  memcpy(data, str.c_str(), str.length());
  data += str.length();
  return *this;
}

auto BufferStream::operator<<(Core::Buffer const& buffer) -> BufferStream& {
  memcpy(data, buffer.data, buffer.size);
  data += buffer.size;
  return *this;
}

MemoryArena::~MemoryArena() {
  FreeAligned(currentBlock);
  for (auto& block : usedBlocks) FreeAligned(block.second);
  for (auto& block : availableBlocks) FreeAligned(block.second);
}

auto MemoryArena::alloc(size_t nBytes) noexcept -> void* {
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
    for (auto iter = availableBlocks.begin(); iter != availableBlocks.end();
         iter++) {
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
    704, 768, 832, 896, 960, 1024};

static const uint32_t kPageSize = 8192;
static const uint32_t kAlignment = 4;

// number of elements in the block size array
static const uint32_t kNumBlockSizes =
    sizeof(kBlockSizes) / sizeof(kBlockSizes[0]);

// largest valid block size
static const uint32_t kMaxBlockSize = kBlockSizes[kNumBlockSizes - 1];

MemoryManager* MemoryManager::singleton = nullptr;

auto MemoryManager::startUp() noexcept -> void {
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

auto MemoryManager::shutDown() noexcept -> void {
  if (pAllocators) delete[] pAllocators;
  if (pBlockSizeLookup) delete[] pBlockSizeLookup;

  singleton = nullptr;
}

auto MemoryManager::allocate(size_t size) noexcept -> void* {
  Allocator* pAlloc = lookUpAllocator(size);
  if (pAlloc)
    return pAlloc->allocate();
  else
    return AllocAligned(size);
}

auto MemoryManager::free(void* p, size_t size) noexcept -> void {
  Allocator* pAlloc = lookUpAllocator(size);
  if (pAlloc)
    pAlloc->free(p);
  else
    FreeAligned(p);
}

auto MemoryManager::lookUpAllocator(size_t size) noexcept -> Allocator* {
  // check eligibility for lookup
  if (size <= kMaxBlockSize)
    return pAllocators + pBlockSizeLookup[size];
  else
    return nullptr;
}

StackAllocator::StackAllocator(size_t size_bytes)
    : capacity(size_bytes), size(0) {
  data = reinterpret_cast<char*>(AllocAligned(capacity));
  top = 0;
}

StackAllocator::~StackAllocator() { FreeAligned((void*)data); }

auto StackAllocator::alloc(size_t size_bytes) noexcept -> void* {
  void* ret = reinterpret_cast<void*>(data[top]);
  if (top + size_bytes > capacity) return nullptr;
  top += size_bytes;
  return ret;
}

auto StackAllocator::getMarker() noexcept -> uint32_t { return uint32_t(top); }

auto StackAllocator::freeToMarker(uint32_t marker) noexcept -> void {
  top = marker;
}

auto StackAllocator::clear() noexcept -> void { top = 0; }
DoubleEndedStackAllocator::DoubleEndedStackAllocator(size_t size_bytes)
    : capacity(size_bytes), size(0) {
  data = new char8_t[capacity];
  top = reinterpret_cast<size_t>(data);
}

DoubleEndedStackAllocator::~DoubleEndedStackAllocator() { delete[] data; }

auto DoubleEndedStackAllocator::alloc(size_t size_bytes) noexcept -> void* {
  void* ret = reinterpret_cast<void*>(top);
  top += size_bytes;
  return ret;
}

auto DoubleEndedStackAllocator::getMarker() noexcept -> void* {
  return reinterpret_cast<void*>(top);
}

auto DoubleEndedStackAllocator::freeToMarker(void* marker) noexcept -> void {
  top = reinterpret_cast<size_t>(marker);
}

auto DoubleEndedStackAllocator::clear() noexcept -> void {
  top = reinterpret_cast<size_t>(data);
}
}