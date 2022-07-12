module;
#include <cstdint>
#include <memory>
export module Core.Memory:Buffer;
import :MemoryManager;

namespace SIByL::Core
{
	export struct Buffer
	{
		Buffer();
		Buffer(size_t size);
		Buffer(Buffer const& b);
		Buffer(Buffer&& b);
		~Buffer();
		auto operator=(Buffer const& b) -> Buffer&;
		auto operator=(Buffer&& b) -> Buffer&;

		auto release() noexcept -> void;

		void* data;
		size_t size;
	};

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
}