module;
#include <cstdint>
export module Core.Memory:StackAllocator;

namespace SIByL::Core
{
	class StackAllocator
	{
	public:
		explicit StackAllocator(size_t size_bytes);
		~StackAllocator();

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

	StackAllocator::StackAllocator(size_t size_bytes)
		: capacity(size_bytes)
		, size(0)
	{
		data = new char8_t[capacity];
		top = reinterpret_cast<size_t>(data);
	}

	StackAllocator::~StackAllocator()
	{
		delete[] data;
	}

	auto StackAllocator::alloc(size_t size_bytes) noexcept -> void*
	{
		void* ret = reinterpret_cast<void*>(top);
		top += size_bytes;
		return ret;
	}

	auto StackAllocator::getMarker() noexcept -> void*
	{
		return reinterpret_cast<void*>(top);
	}

	auto StackAllocator::freeToMarker(void* marker) noexcept -> void
	{
		top = reinterpret_cast<size_t>(marker);
	}

	auto StackAllocator::clear() noexcept -> void
	{
		top = reinterpret_cast<size_t>(data);
	}

}