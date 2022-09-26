module;
#include <cstdint>
#include <memory>
#include <string>
export module Core.Memory:Buffer;
import :MemoryManager;

namespace SIByL::Core
{
	struct BufferStream;

	export struct Buffer {
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

	export struct BufferStream
	{
		char* data;

		auto operator<<(char c) -> BufferStream&;
		auto operator<<(std::string const& string) -> BufferStream&;
		auto operator<<(Core::Buffer const& buffer) -> BufferStream&;
	};
}