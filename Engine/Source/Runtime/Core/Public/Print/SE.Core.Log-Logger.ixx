module;
#include <cstdint>
#include <memory>
#include <ostream>
#include <chrono>
#include <format>
#include <iostream>
#include <string>
#include <functional>
export module SE.Core.Log:Logger;
import SE.Core.System;
import :LogStream;

namespace SIByL::Core
{
	/** Logger helps with log different kinds of information */
	export struct Logger {
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

#pragma region LOGGER_IMPL

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

	inline auto Logger::clear() noexcept -> void {
		stream.block.clear();
	}

	inline auto Logger::pushTimestamp(char const* type) noexcept -> void {
		std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
		std::time_t now_time = std::chrono::system_clock::to_time_t(now);
		std::tm tm;
		localtime_s(&tm, &now_time);
		if (type != nullptr) stream << type;
		stream << std::format("[{:02d}:{:02d}:{:02d}] ", tm.tm_hour, tm.tm_min, tm.tm_min).c_str();
	}

#pragma endregion
}