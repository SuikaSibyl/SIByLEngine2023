module;
#include <cstdint>
#include <memory>
#include <ostream>
export module SE.Core.Log:LogStream;

namespace SIByL::Core
{
	/** A block of memory to contain data in a log stream. */
	struct LogStreamBlock {
		/** default constructor */
		LogStreamBlock();
		LogStreamBlock(LogStreamBlock &&)		= delete;
		LogStreamBlock(LogStreamBlock const&)	= delete;
		auto operator=(LogStreamBlock &&)		->LogStreamBlock const& = delete;
		auto operator=(LogStreamBlock const&)	->LogStreamBlock const& = delete;
		~LogStreamBlock();
		/** request some data from back of the buffer.
		* if it exceed the capacity, it would automatically do expand. */
		inline auto alloc(size_t size) noexcept -> void*;
		/** expand the capacity twice as large as current size */
		inline auto expand() noexcept -> void*;
		/** clear the data buffer, but do not release the buffer */
		inline auto clear() noexcept -> void*;
		/** current used buffer size */
		size_t size = 0;
		/** current allocated buffer size */
		size_t capacity = 0;
		/** data buffer */
		char8_t* data = nullptr;
	};

	/** A stream to contain log info as iostream. */
	export struct LogStream {
		/** get the current written size of data */
		inline auto getSize() noexcept -> size_t { return block.size; }
		/** output the logstream to ostream */
		friend inline auto operator<<(std::ostream& os, LogStream& ls) -> std::ostream&;
		/** output the char to logstream */
		friend inline auto operator<<(LogStream& os, char const* c) -> LogStream&;
		/** output anything else to logstream */
		template <class T> friend inline auto operator<<(LogStream& os, T const& c) -> LogStream&;
		/** operators to move cursor to each direction */
		struct cursor_left	{ uint32_t n; };
		struct cursor_right	{ uint32_t n; };
		struct cursor_up	{ uint32_t n; };
		struct cursor_down	{ uint32_t n; };
		/** operators to change color of letters */
		static char const* white;
		static char const* black;
		static char const* red;
		static char const* green;
		static char const* yellow;
		static char const* blue;
		static char const* purple;
		static char const* cyan;
		/** block to contain the data memory */
		LogStreamBlock block;
	};

#pragma region LOG_STREAM_BLOCK_IMPL

	LogStreamBlock::LogStreamBlock() {
		size = 0;
		capacity = 8192;
		data = new char8_t[8192];
	}

	LogStreamBlock::~LogStreamBlock() {
		delete[] data;
	}

	inline auto LogStreamBlock::alloc(size_t s) noexcept -> void* {
		size_t const space_need = size + s + 1;
		while (space_need > capacity)
			expand();
		size += s;
		return data + size - s;
	}

	inline auto LogStreamBlock::expand() noexcept -> void* {
		capacity *= 2;
		char8_t* data = new char8_t[capacity];
		memcpy(data, this->data, size);
		std::swap(data, this->data);
		delete[] data;
		return this->data + size;
	}

	inline auto LogStreamBlock::clear() noexcept -> void* {
		size = 0;
		return data;
	}

#pragma endregion

#pragma region LOG_STREAM_IMPL

	char const* LogStream::white = "\033[0m";
	char const* LogStream::black = "\033[0;30m";
	char const* LogStream::red = "\033[0;31m";
	char const* LogStream::green = "\033[0;32m";
	char const* LogStream::yellow = "\033[0;33m";
	char const* LogStream::blue = "\033[0;34m";
	char const* LogStream::purple = "\033[0;35m";
	char const* LogStream::cyan = "\033[0;36m";

	export inline auto operator<<(std::ostream& os, LogStream& ls) -> std::ostream& {
		ls.block.data[ls.block.size] = '\0';
		os << (char*)ls.block.data;
		return os;
	}

	export inline auto operator<<(LogStream& os, char const* c) -> LogStream& {
		size_t lenth = strlen(c);
		void* begin = os.block.alloc(lenth);
		memcpy(begin, c, lenth);
		return os;
	}

	export template <class T>
	inline auto operator<<(LogStream& os, T const& c) -> LogStream& {
		void* begin = os.block.alloc(0);
		size_t valid_length = os.block.capacity - os.block.size - 1;
		size_t const length = snprintf(reinterpret_cast<char*>(begin), valid_length, "%d", c);
		while (length < 0 || length > valid_length) {
			begin = os.block.expand();
			valid_length = os.block.capacity - os.block.size - 1;
			snprintf(reinterpret_cast<char*>(begin), os.block.capacity - os.block.size - 1, "%d", c);
		}
		os.block.alloc(length);
		return os;
	}

	export template <>
	inline auto operator<<(LogStream& os, float const& c) -> LogStream& {
		void* begin = os.block.alloc(0);
		size_t valid_length = os.block.capacity - os.block.size - 1;
		size_t const length = snprintf(reinterpret_cast<char*>(begin), valid_length, "%f", c);
		while (length < 0 || length > valid_length) {
			begin = os.block.expand();
			valid_length = os.block.capacity - os.block.size - 1;
			snprintf(reinterpret_cast<char*>(begin), os.block.capacity - os.block.size - 1, "%f", c);
		}
		os.block.alloc(length);
		return os;
	}

	export template <>
	inline auto operator<<(LogStream& os, double const& c) -> LogStream& {
		void* begin = os.block.alloc(0);
		size_t valid_length = os.block.capacity - os.block.size - 1;
		size_t const length = snprintf(reinterpret_cast<char*>(begin), os.block.capacity - os.block.size - 1, "%lf", c);
		while (length < 0 || length > valid_length) {
			begin = os.block.expand();
			valid_length = os.block.capacity - os.block.size - 1;
			snprintf(reinterpret_cast<char*>(begin), os.block.capacity - os.block.size - 1, "%lf", c);
		}
		os.block.alloc(length);
		return os;
	}

	export template <>
	inline auto operator<<(LogStream& os, LogStream::cursor_left const& c) -> LogStream& {
		os << "\033[" << c.n << "D";
		return os;
	}

	export template <>
	inline auto operator<<(LogStream& os, LogStream::cursor_right const& c) -> LogStream& {
		os << "\033[" << c.n << "C";
		return os;
	}

	export template <>
	inline auto operator<<(LogStream& os, LogStream::cursor_up const& c) -> LogStream& {
		os << "\033[" << c.n << "A";
		return os;
	}

	export template <>
	inline auto operator<<(LogStream& os, LogStream::cursor_down const& c) -> LogStream& {
		os << "\033[" << c.n << "B";
		return os;
	}

#pragma endregion
}