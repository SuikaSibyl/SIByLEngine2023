/**
 * ----------------------------------------
 * EXAMPLES
 * ----------------------------------------
 * Users should call predefined static functions in LogManager to log a message.
 * Pass a string to logger, construct it with std::format if necessary.
 * Logger will attach a timestamp in the beginning and change the color
 * according to log type.
 *
 * ```
 * Core::LogManager::Debug(some_string);		\\ output a debug-type
 * log, use cyan color
 * Core::LogManager::Log(some_string);		\\ output a log-type	 log,
 * use white color Core::LogManager::Warning(some_string);	\\ output a
 * warning-type log, use yellow color Core::LogManager::Error(some_string);
 * \\ output a error-type	 log, use red color
 * ```
 */

#pragma once
#include <chrono>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

#include <System/SE.Core.System.hpp>

namespace SIByL::Core {
/** A block of memory to contain data in a log stream. */
struct LogStreamBlock {
  /** default constructor */
  LogStreamBlock();
  LogStreamBlock(LogStreamBlock&&) = delete;
  LogStreamBlock(LogStreamBlock const&) = delete;
  auto operator=(LogStreamBlock&&) -> LogStreamBlock const& = delete;
  auto operator=(LogStreamBlock const&) -> LogStreamBlock const& = delete;
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
SE_EXPORT struct LogStream {
  /** get the current written size of data */
  inline auto getSize() noexcept -> size_t { return block.size; }
  /** output the logstream to ostream */
  friend inline auto operator<<(std::ostream& os, LogStream& ls)
      -> std::ostream&;
  /** output the char to logstream */
  friend inline auto operator<<(LogStream& os, char const* c) -> LogStream&;
  /** output anything else to logstream */
  template <class T>
  friend inline auto operator<<(LogStream& os, T const& c) -> LogStream&;
  /** operators to move cursor to each direction */
  struct cursor_left {
    uint32_t n;
  };
  struct cursor_right {
    uint32_t n;
  };
  struct cursor_up {
    uint32_t n;
  };
  struct cursor_down {
    uint32_t n;
  };
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
SE_EXPORT struct Logger {
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
/** Log manager is a singleton that helps with logging */
SE_EXPORT struct LogManager : public Manager {
  /** start up the manager */
  virtual auto startUp() noexcept -> void override;
  /** shut down the manager */
  virtual auto shutDown() noexcept -> void override;
  /** get the singleton */
  static inline auto get() noexcept -> LogManager* { return singleton; }
  /** get the global logger */
  static inline auto getLogger() noexcept -> Logger& { return get()->logger; }
  /** until to log a debug log */
  static inline auto Debug(std::string const& s) noexcept -> void;
  /** until to log a log log */
  static inline auto Log(std::string const& s) noexcept -> void;
  /** until to log a warning log */
  static inline auto Warning(std::string const& s) noexcept -> void;
  /** until to log a error log */
  static inline auto Error(std::string const& s) noexcept -> void;
  /** until to log a custom log */
  static inline auto Correct(std::string const& s) noexcept -> void;
  /** a callback func for editor to get logged info */
  std::function<void(std::string const&)> editorCallback = nullptr;

 private:
  /** the global logger */
  Logger logger;
  /** the singleton of LogManager */
  static LogManager* singleton;
};

inline char const* LogStream::white = "\033[0m";
inline char const* LogStream::black = "\033[0;30m";
inline char const* LogStream::red = "\033[0;31m";
inline char const* LogStream::green = "\033[0;32m";
inline char const* LogStream::yellow = "\033[0;33m";
inline char const* LogStream::blue = "\033[0;34m";
inline char const* LogStream::purple = "\033[0;35m";
inline char const* LogStream::cyan = "\033[0;36m";

SE_EXPORT inline auto operator<<(std::ostream& os, LogStream& ls)
    -> std::ostream& {
  ls.block.data[ls.block.size] = '\0';
  os << (char*)ls.block.data;
  return os;
}

SE_EXPORT inline auto operator<<(LogStream& os, char const* c) -> LogStream& {
  size_t lenth = strlen(c);
  void* begin = os.block.alloc(lenth);
  memcpy(begin, c, lenth);
  return os;
}

SE_EXPORT template <class T>
inline auto operator<<(LogStream& os, T const& c) -> LogStream& {
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

SE_EXPORT template <>
inline auto operator<<(LogStream& os, float const& c) -> LogStream& {
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

SE_EXPORT template <>
inline auto operator<<(LogStream& os, double const& c) -> LogStream& {
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

SE_EXPORT template <>
inline auto operator<<(LogStream& os, LogStream::cursor_left const& c)
    -> LogStream& {
  os << "\033[" << c.n << "D";
  return os;
}

SE_EXPORT template <>
inline auto operator<<(LogStream& os, LogStream::cursor_right const& c)
    -> LogStream& {
  os << "\033[" << c.n << "C";
  return os;
}

SE_EXPORT template <>
inline auto operator<<(LogStream& os, LogStream::cursor_up const& c)
    -> LogStream& {
  os << "\033[" << c.n << "A";
  return os;
}

SE_EXPORT template <>
inline auto operator<<(LogStream& os, LogStream::cursor_down const& c)
    -> LogStream& {
  os << "\033[" << c.n << "B";
  return os;
}

inline auto LogManager::Debug(std::string const& s) noexcept -> void {
#ifdef _NEED_LOG
  getLogger().debug(s);
  if (singleton->editorCallback) singleton->editorCallback("[D]" + s);
#endif
}

inline auto LogManager::Log(std::string const& s) noexcept -> void {
#ifdef _NEED_LOG
  getLogger().log(s);
  if (singleton->editorCallback) singleton->editorCallback("[L]" + s);
#endif
}

inline auto LogManager::Warning(std::string const& s) noexcept -> void {
#ifdef _NEED_LOG
  getLogger().warning(s);
  if (singleton->editorCallback) singleton->editorCallback("[W]" + s);
#endif
}

inline auto LogManager::Error(std::string const& s) noexcept -> void {
#ifdef _NEED_LOG
  getLogger().error(s);
  if (singleton->editorCallback) singleton->editorCallback("[E]" + s);
#endif
}

inline auto LogManager::Correct(std::string const& s) noexcept -> void {
#ifdef _NEED_LOG
  getLogger().correct(s);
  if (singleton->editorCallback) singleton->editorCallback("[C]" + s);
#endif
}
}  // namespace SIByL::Core


namespace SIByL::Core {
#pragma region LOGGER_IMPL

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

#pragma endregion
}  // namespace SIByL::Core
