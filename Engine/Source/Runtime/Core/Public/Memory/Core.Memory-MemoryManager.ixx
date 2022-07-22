module;
#include <cstdint>
#include <utility>
#include <memory>
#include <type_traits>
export module Core.Memory:MemoryManager;
import Core.System;
import :Allocator;

namespace SIByL::Core
{
	/**
	* Memory Manager mainly use a set of Allocators to allocate memory from specific allocator.
	* It is a Manager struct, whose singleton is store in Application::Root.
	* @see Allocator
	*/
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
}