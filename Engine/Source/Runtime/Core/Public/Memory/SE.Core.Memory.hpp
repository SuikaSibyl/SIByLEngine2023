#pragma once
#include <new>
#include <list>
#include <string>
#include <cstdint>
#include <utility>
#include <memory>
#include <type_traits>
#include <System/SE.Core.System.hpp>

namespace SIByL::Core {
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

SE_EXPORT struct Allocator {
  Allocator();
  Allocator(size_t data_size, size_t page_size, size_t alignment);
  ~Allocator();

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
/**
 * Two advantages:
 * 1. allocation is extremely fast, usually just a pointer increment
 * 2. improve locality of reference and lead to fewer cache misses
 */
SE_EXPORT struct MemoryArena {
  /* Allocate in memory chunks in size of blockSize, which is defaultly 256kB */
  MemoryArena(size_t blockSize = 262144) : blockSize(blockSize) {}
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
    for (size_t i = 0; i < n; i++) new (&ret[i]) T();
  return ret;
}

SE_EXPORT template <typename T, typename... Args>
inline auto ArenaAlloc(MemoryArena& arena, Args&&... args) noexcept -> T* {
  return ::new (arena.alloc(sizeof(T))) T(std::forward<Args>(args)...);
}

/**
 * Memory Manager mainly use a set of Allocators to allocate memory from
 * specific allocator. It is a Manager struct, whose singleton is store in
 * Application::Root.
 * @see Allocator
 */
SE_EXPORT struct MemoryManager : public Manager {
  virtual auto startUp() noexcept -> void override;
  virtual auto shutDown() noexcept -> void override;

  auto allocate(size_t size) noexcept -> void*;
  auto free(void* p, size_t size) noexcept -> void;
  static auto get() noexcept -> MemoryManager* { return singleton; }

 private:
  size_t* pBlockSizeLookup;
  Allocator* pAllocators;
  static MemoryManager* singleton;
  auto lookUpAllocator(size_t size) noexcept -> Allocator*;
};

/**
 * Could not be released in random order, but need to be in reversed order of
 * allocation.
 */
class StackAllocator {
 public:
  explicit StackAllocator(size_t size_bytes);
  ~StackAllocator();
  /** alloca a new block on the top of stack */
  auto alloc(size_t size_bytes) noexcept -> void*;
  /** get the marker pointing to the top of the stack */
  auto getMarker() noexcept -> uint32_t;
  /** revert the stack to a former marker */
  auto freeToMarker(uint32_t marker) noexcept -> void;
  /** revert the stack to zero marker */
  auto clear() noexcept -> void;
 private:
  size_t capacity;
  size_t size;
  size_t top;
  char* data;
};

struct DoubleEndedStackAllocator {
  explicit DoubleEndedStackAllocator(size_t size_bytes);
  ~DoubleEndedStackAllocator();
  auto alloc(size_t size_bytes) noexcept -> void*;
  auto getMarker() noexcept -> void*;
  auto freeToMarker(void* marker) noexcept -> void;
  auto clear() noexcept -> void;
 private:
  size_t capacity;
  size_t size;
  void* data;
  size_t top;
};
struct BufferStream;

SE_EXPORT struct Buffer {
  Buffer();
  Buffer(size_t size);
  Buffer(Buffer const& b);
  Buffer(Buffer&& b);
  ~Buffer();
  auto operator=(Buffer const& b) -> Buffer&;
  auto operator=(Buffer&& b) -> Buffer&;

  auto release() noexcept -> void;
  auto stream() noexcept -> BufferStream;

  void* data = nullptr;
  size_t size = 0;
};

SE_EXPORT struct BufferStream {
  char* data;

  auto operator<<(char c) -> BufferStream&;
  auto operator<<(std::string const& string) -> BufferStream&;
  auto operator<<(Core::Buffer const& buffer) -> BufferStream&;
};
/**
 * 2D array with better spatial memory locality.
 * The 2D array is represented by square blocks of a small fixed size that is a
 * power of 2.
 */
SE_EXPORT template <class T, int logBlockSize = 2>
struct BlockedArray {
  /** @para d: a standard c++ array to initialize, which is optional */
  BlockedArray(int uRes, int vRes, T const* d = nullptr);

  /** access each element in blocked array */
  auto operator()(int u, int v) -> T&;

  /** size of a single block */
  constexpr auto blockSize() const noexcept -> int;
  /** round both dimensions up to be a multiple of the block size */
  auto roundUp(int x) const noexcept -> int;

  auto uSize() const noexcept -> int { return uRes; }
  auto vSize() const noexcept -> int { return vRes; }

  auto block(int a) const noexcept -> int { return a >> logBlockSize; }
  auto offset(int a) const noexcept -> int { return (a & (blockSize() - 1)); }

 private:
  T* data;
  int const uRes, vRes, uBlocks;
};

template <class T, int logBlockSize>
BlockedArray<T, logBlockSize>::BlockedArray(int uRes, int vRes, T const* d)
    : uRes(uRes), vRes(vRes), uBlocks(roundUp(uRes) >> logBlockSize) {
  int nAlloc = roundUp(uRes) * roundUp(vRes);
  data = AllocAligned<T>(nAlloc);
  for (int i = 0; i < nAlloc; ++i) new (&data[i]) T();
  if (d)
    for (int v = 0; v < vRes; v++)
      for (int u = 0; u < uRes; u++) (*this)(u, v) = d[v * uRes + u];
}

template <class T, int logBlockSize>
constexpr auto BlockedArray<T, logBlockSize>::blockSize() const noexcept
    -> int {
  return 1 << logBlockSize;
}

template <class T, int logBlockSize>
auto BlockedArray<T, logBlockSize>::roundUp(int x) const noexcept -> int {
  return (x + blockSize() - 1) & ~(blockSize() - 1);
}

template <class T, int logBlockSize>
auto BlockedArray<T, logBlockSize>::operator()(int u, int v) -> T& {
  int bu = block(u), bv = block(v);
  int ou = offset(u), ov = offset(v);
  int offset = blockSize() * blockSize() * (uBlocks * bv + bu);
  offset += blockSize() * ov + ou;
  return data[offset];
}

#define L1_CACHE_LINE_SIZE 64

/**
 * Alloca could allocate memory from system stack, which is useful for scoped
 * new objects. The memory allocated by Alloca will be automatically freed after
 * scope is end.
 */

// However Alloca could not be implemented/warpped as function, as it allocate
// memory on stack, which would be release right after the function returns, so
// the memory will be invalid.

/**
 * Aligned Allocation & Deallocation could query aligned memory, the impl is
 * platform specific.
 * @todo Only Windows is supported in current version
 */
SE_EXPORT inline auto AllocAligned(size_t size) -> void* {
  return _aligned_malloc(size, L1_CACHE_LINE_SIZE);
}

SE_EXPORT template <class T>
inline auto AllocAligned(size_t count) -> T* {
  return (T*)AllocAligned(count * sizeof(T));
}

SE_EXPORT inline auto FreeAligned(void* p) -> void { _aligned_free(p); }

/**
 * Custom Alloc/Free/New/Delete functions allocate from MemoryManager singleton.
 * No frequent system new/delete is need, which brings speed-up.
 */
SE_EXPORT inline auto Alloc(size_t size) -> void* {
  return MemoryManager::get()->allocate(size);
}

SE_EXPORT inline auto Free(void* p, size_t size) -> void {
  return MemoryManager::get()->free(p, size);
}

SE_EXPORT template <typename T, typename... Args>
inline auto New(Args&&... args) noexcept -> T* {
  return ::new (MemoryManager::get()->allocate(sizeof(T)))
      T(std::forward<Args>(args)...);
}

SE_EXPORT template <typename T>
inline auto Delete(T* p) noexcept -> void {
  reinterpret_cast<T*>(p)->~T();
  MemoryManager::get()->free(p, sizeof(T));
}
}  // namespace SIByL::Core