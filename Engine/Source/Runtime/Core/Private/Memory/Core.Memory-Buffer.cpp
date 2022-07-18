module;
#include <cstdint>
#include <memory>
#include <string>
module Core.Memory:Buffer;
import Core.Memory;
import :MemoryManager;

namespace SIByL::Core
{
	Buffer::Buffer()
		:data(nullptr), size(0)
	{}

	Buffer::Buffer(size_t size)
		:size(size)
	{
		data = Alloc(size);
	}

	Buffer::Buffer(Buffer const& b)
	{
		release();
		size = b.size;
		data = Alloc(size);
		memcpy(data, b.data, size);
	}

	Buffer::Buffer(Buffer&& b)
	{
		release();
		size = b.size;
		data = b.data;
		b.data = nullptr;
		b.size = 0;
	}

	Buffer::~Buffer()
	{
		release();
	}

	auto Buffer::operator=(Buffer const& b)->Buffer&
	{
		release();
		size = b.size;
		data = Alloc(size);
		memcpy(data, b.data, size);
		return *this;
	}

	auto Buffer::operator=(Buffer&& b)->Buffer&
	{
		release();
		size = b.size;
		data = b.data;
		b.data = nullptr;
		b.size = 0;
		return *this;
	}

	auto Buffer::release() noexcept -> void
	{
		if (data == nullptr) return;
		Free(data, size);
		data = nullptr;
		size = 0;
	}

	auto Buffer::stream() noexcept -> BufferStream
	{
		return BufferStream{ reinterpret_cast<char*>(data) };
	}

	auto BufferStream::operator<<(char c) -> BufferStream&
	{
		data[0] = c;
		data++;
		return *this;
	}

	auto BufferStream::operator<<(std::string const& str) -> BufferStream&
	{
		memcpy(data, str.c_str(), str.length());
		data += str.length();
		return *this;
	}

	auto BufferStream::operator<<(Core::Buffer const& buffer)->BufferStream&
	{
		memcpy(data, buffer.data, buffer.size);
		data += buffer.size;
		return *this;
	}
}