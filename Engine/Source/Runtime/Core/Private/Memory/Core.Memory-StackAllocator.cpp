module;
#include <cstdint>
module Core.Memory:StackAllocator;
import Core.Memory;

namespace SIByL::Core
{
	StackAllocator::StackAllocator(size_t size_bytes)
		: capacity(size_bytes)
		, size(0)
	{
		data = reinterpret_cast<char*>(AllocAligned(capacity));
		top = 0;
	}

	StackAllocator::~StackAllocator() {
		FreeAligned((void*)data);
	}

	auto StackAllocator::alloc(size_t size_bytes) noexcept -> void* {
		void* ret = reinterpret_cast<void*>(data[top]);
		if (top + size_bytes > capacity) return nullptr;
		top += size_bytes;
		return ret;
	}

	auto StackAllocator::getMarker() noexcept -> uint32_t {
		return top;
	}

	auto StackAllocator::freeToMarker(uint32_t marker) noexcept -> void {
		top = marker;
	}

	auto StackAllocator::clear() noexcept -> void {
		top = 0;
	}

}