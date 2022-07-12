module;
#include <cstdint>
#include <utility>
#include <type_traits>
export module Core.Memory:MemoryManager;
import Core.System;
import :Allocator;

namespace SIByL::Core
{
	export struct MemoryManager :public Manager
	{
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
	
	MemoryManager* MemoryManager::singleton = nullptr;

	export inline auto Alloc(size_t size) -> void*
	{
		return MemoryManager::get()->allocate(size);
	}

	export inline auto Free(void* p, size_t size) -> void
	{
		return MemoryManager::get()->free(p, size);
	}

	export template<typename T, typename... Args>
		inline auto New(Args&&... args) noexcept -> T*
	{
		return ::new (MemoryManager::get()->allocate(sizeof(T))) T(std::forward<Args>(args)...);
	}

	export template<typename T>
		inline auto Delete(T* p) noexcept -> void
	{
		reinterpret_cast<T*>(p)->~T();
		MemoryManager::get()->free(p, sizeof(T));
	}
}