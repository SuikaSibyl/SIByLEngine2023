module;
#include <cstdint>
export module SE.Core.Memory:DoubleEndedStackAllocator;

namespace SIByL::Core
{
	class DoubleEndedStackAllocator
	{
	public:
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

	DoubleEndedStackAllocator::DoubleEndedStackAllocator(size_t size_bytes)
		: capacity(size_bytes)
		, size(0)
	{
		data = new char8_t[capacity];
		top = reinterpret_cast<size_t>(data);
	}

	DoubleEndedStackAllocator::~DoubleEndedStackAllocator()
	{
		delete[] data;
	}

	auto DoubleEndedStackAllocator::alloc(size_t size_bytes) noexcept -> void*
	{
		void* ret = reinterpret_cast<void*>(top);
		top += size_bytes;
		return ret;
	}

	auto DoubleEndedStackAllocator::getMarker() noexcept -> void*
	{
		return reinterpret_cast<void*>(top);
	}

	auto DoubleEndedStackAllocator::freeToMarker(void* marker) noexcept -> void
	{
		top = reinterpret_cast<size_t>(marker);
	}

	auto DoubleEndedStackAllocator::clear() noexcept -> void
	{
		top = reinterpret_cast<size_t>(data);
	}

}