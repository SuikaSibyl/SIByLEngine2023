#define DLIB_EXPORT
#include <se.core.hpp>
#undef DLIB_EXPORT
#include <chrono>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

namespace se {
/** A block of memory to contain data in a log stream. */
struct LogStreamBlock {
	/** default constructor */
	LogStreamBlock();
	LogStreamBlock(LogStreamBlock&&) = delete;
	LogStreamBlock(LogStreamBlock const&) = delete;
	auto operator=(LogStreamBlock&&)->LogStreamBlock const& = delete;
	auto operator=(LogStreamBlock const&)->LogStreamBlock const& = delete;
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
struct LogStream {
	/** get the current written size of data */
	inline auto getSize() noexcept -> size_t { return block.size; }
	/** output the logstream to ostream */
	friend inline auto operator<<(std::ostream& os, LogStream& ls)->std::ostream&;
	/** output the char to logstream */
	friend inline auto operator<<(LogStream& os, char const* c)->LogStream&;
	/** output anything else to logstream */
	template <class T>
	friend inline auto operator<<(LogStream& os, T const& c)->LogStream&;
	/** operators to move cursor to each direction */
	struct cursor_left	{ uint32_t n; };
	struct cursor_right { uint32_t n; };
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

/** Logger helps with log different kinds of information */
struct Logger {
	/* is a singleton */
	SINGLETON(Logger, {});
	/* begin a log, pushing the timestamp */
	inline auto beginLog(char const* type = nullptr) noexcept -> LogStream&;
	/* print the log to std::cout */
	inline auto flushLog() noexcept -> void;
	/** clear the stream block */
	inline auto clear() noexcept -> void;
	/** push a timestamp to the stream */
	inline auto pushTimestamp(char const* type = nullptr) noexcept -> void;
	// predefined log types
	// -----------------------------------
	/** predefined log type, use white color */
	inline auto log(std::string const& s) noexcept -> void;
	/** predefined log type, use cyan color */
	inline auto debug(std::string const& s) noexcept -> void;
	/** predefined log type, use yellow color */
	inline auto warning(std::string const& s) noexcept -> void;
	/** predefined log type, use red color */
	inline auto error(std::string const& s) noexcept -> void;
	/** predefined log type, use custom color */
	inline auto correct(std::string const& s) noexcept -> void;
	/** stream for the logger */
	LogStream stream;
};

inline char const* LogStream::white = "\033[0m";
inline char const* LogStream::black = "\033[0;30m";
inline char const* LogStream::red = "\033[0;31m";
inline char const* LogStream::green = "\033[0;32m";
inline char const* LogStream::yellow = "\033[0;33m";
inline char const* LogStream::blue = "\033[0;34m";
inline char const* LogStream::purple = "\033[0;35m";
inline char const* LogStream::cyan = "\033[0;36m";

inline auto operator<<(std::ostream& os, LogStream& ls) -> std::ostream& {
  ls.block.data[ls.block.size] = '\0';
  os << (char*)ls.block.data;
  return os;
}

inline auto operator<<(LogStream& os, char const* c) -> LogStream& {
  size_t lenth = strlen(c);
  void* begin = os.block.alloc(lenth);
  memcpy(begin, c, lenth);
  return os;
}

template <class T> inline auto operator<<(LogStream& os, T const& c) -> LogStream& {
  void* begin = os.block.alloc(0);
  size_t valid_length = os.block.capacity - os.block.size - 1;
  size_t const length =
      snprintf(reinterpret_cast<char*>(begin), valid_length, "%d", c);
  while (length < 0 || length > valid_length) {
    begin = os.block.expand();
    valid_length = os.block.capacity - os.block.size - 1;
    snprintf(reinterpret_cast<char*>(begin),
             os.block.capacity - os.block.size - 1, "%d", c);
  }
  os.block.alloc(length);
  return os;
}

template <> inline auto operator<<(LogStream& os, float const& c) -> LogStream& {
  void* begin = os.block.alloc(0);
  size_t valid_length = os.block.capacity - os.block.size - 1;
  size_t const length =
      snprintf(reinterpret_cast<char*>(begin), valid_length, "%f", c);
  while (length < 0 || length > valid_length) {
    begin = os.block.expand();
    valid_length = os.block.capacity - os.block.size - 1;
    snprintf(reinterpret_cast<char*>(begin),
             os.block.capacity - os.block.size - 1, "%f", c);
  }
  os.block.alloc(length);
  return os;
}

template <> inline auto operator<<(LogStream& os, double const& c) -> LogStream& {
  void* begin = os.block.alloc(0);
  size_t valid_length = os.block.capacity - os.block.size - 1;
  size_t const length =
      snprintf(reinterpret_cast<char*>(begin),
               os.block.capacity - os.block.size - 1, "%lf", c);
  while (length < 0 || length > valid_length) {
    begin = os.block.expand();
    valid_length = os.block.capacity - os.block.size - 1;
    snprintf(reinterpret_cast<char*>(begin),
             os.block.capacity - os.block.size - 1, "%lf", c);
  }
  os.block.alloc(length);
  return os;
}

template <> inline auto operator<<(LogStream& os, LogStream::cursor_left const& c)
    -> LogStream& {
  os << "\033[" << c.n << "D";
  return os;
}

template <> inline auto operator<<(LogStream& os, LogStream::cursor_right const& c)
    -> LogStream& {
  os << "\033[" << c.n << "C";
  return os;
}

template <> inline auto operator<<(LogStream& os, LogStream::cursor_up const& c)
    -> LogStream& {
  os << "\033[" << c.n << "A";
  return os;
}

template <> inline auto operator<<(LogStream& os, LogStream::cursor_down const& c)
    -> LogStream& {
  os << "\033[" << c.n << "B";
  return os;
}

inline auto LogStreamBlock::alloc(size_t s) noexcept -> void* {
  size_t const space_need = size + s + 1;
  while (space_need > capacity) expand();
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

inline auto Logger::beginLog(char const* type) noexcept -> LogStream& {
  clear();
  pushTimestamp(type);
  return stream;
}

inline auto Logger::flushLog() noexcept -> void {
  std::cout << stream << std::endl;
}

inline auto Logger::log(std::string const& s) noexcept -> void {
  auto& stream = beginLog();
  stream << s.c_str();
  flushLog();
}

inline auto Logger::debug(std::string const& s) noexcept -> void {
  auto& stream = beginLog(LogStream::cyan);
  stream << LogStream::cyan << s.c_str() << LogStream::white;
  flushLog();
}

inline auto Logger::warning(std::string const& s) noexcept -> void {
  auto& stream = beginLog(LogStream::yellow);
  stream << LogStream::yellow << s.c_str() << LogStream::white;
  flushLog();
}

inline auto Logger::error(std::string const& s) noexcept -> void {
  auto& stream = beginLog(LogStream::red);
  stream << LogStream::red << s.c_str() << LogStream::white;
  flushLog();
}

inline auto Logger::correct(std::string const& s) noexcept -> void {
  auto& stream = beginLog(LogStream::green);
  stream << LogStream::green << s.c_str() << LogStream::white;
  flushLog();
}

inline auto Logger::clear() noexcept -> void { stream.block.clear(); }

inline auto Logger::pushTimestamp(char const* type) noexcept -> void {
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm tm;
  localtime_s(&tm, &now_time);
  if (type != nullptr) stream << type;
  stream << std::format("[{:02d}:{:02d}:{:02d}] ", tm.tm_hour, tm.tm_min,
                        tm.tm_min)
                .c_str();
}

LogStreamBlock::LogStreamBlock() {
  size = 0;
  capacity = 8192;
  data = new char8_t[8192];
}

LogStreamBlock::~LogStreamBlock() { delete[] data; }

ex::delegate<void(std::string const&, int type)> root::print::callbacks;

auto root::print::debug(std::string const& s) noexcept -> void {
	Singleton<Logger>::instance()->debug(s);
	if (callbacks) callbacks(s, 0);
}

auto root::print::log(std::string const& s) noexcept -> void {
	Singleton<Logger>::instance()->log(s);
	if (callbacks) callbacks(s, 1);
}

auto root::print::warning(std::string const& s) noexcept -> void {
	Singleton<Logger>::instance()->warning(s);
	if (callbacks) callbacks(s, 2);
}

auto root::print::error(std::string const& s) noexcept -> void {
	Singleton<Logger>::instance()->error(s);
	if (callbacks) callbacks(s, 3);
}

auto root::print::correct(std::string const& s) noexcept -> void {
	Singleton<Logger>::instance()->correct(s);
	if (callbacks) callbacks(s, 4);
}
}
