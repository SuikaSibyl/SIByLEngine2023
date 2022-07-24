module;
#include <cstdint>
export module Core.Memory:StackAllocator;

namespace SIByL::Core
{
	/**
	* Could not be released in random order, but need to be in reversed order of allocation.
	*/
	class StackAllocator
	{
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
}