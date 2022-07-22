module;
#include <cstdint>
#include <utility>
#include <memory>
#include <type_traits>
export module Core.Memory:Memory;
import Core.System;
import :MemoryManager;

#define L1_CACHE_LINE_SIZE 64

namespace SIByL::Core
{
	/**
	* Alloca could allocate memory from system stack, which is useful for scoped new objects.
	* The memory allocated by Alloca will be automatically freed after scope is end.
	*/
	export template<class T>
	inline auto Alloca() noexcept -> T* { return (T*)alloca(sizeof(T)); }

	export template<class T>
	inline auto Alloca(size_t count) noexcept -> T* { return (T*)alloca(count * sizeof(T)); }

	/**
	* Aligned Allocation & Deallocation could query aligned memory, the impl is platform specific.
	* @todo Only Windows is supported in current version
	*/
	export inline auto AllocAligned(size_t size) -> void* {
		return _aligned_malloc(size, L1_CACHE_LINE_SIZE);
	}

	export template<class T>
	inline auto AllocAligned(size_t count) -> T* {
		return (T*)AllocAligned(count * sizeof(T));
	}

	export inline auto FreeAligned(void* p) -> void {
		_aligned_free(p);
	}

	/**
	* Custom Alloc/Free/New/Delete functions allocate from MemoryManager singleton.
	* No frequent system new/delete is need, which brings speed-up.
	*/
	export inline auto Alloc(size_t size) -> void* {
		return MemoryManager::get()->allocate(size);
	}

	export inline auto Free(void* p, size_t size) -> void {
		return MemoryManager::get()->free(p, size);
	}

	export template<typename T, typename... Args>
	inline auto New(Args&&... args) noexcept -> T* {
		return ::new (MemoryManager::get()->allocate(sizeof(T))) T(std::forward<Args>(args)...);
	}

	export template<typename T>
	inline auto Delete(T* p) noexcept -> void {
		reinterpret_cast<T*>(p)->~T();
		MemoryManager::get()->free(p, sizeof(T));
	}
}