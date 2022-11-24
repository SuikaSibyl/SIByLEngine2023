module;
#include <cstdint>
#include <memory>
#include <ostream>
#include <chrono>
#include <format>
#include <iostream>
#include <string>
#include <functional>
export module Core.Log:Logger;
import Core.System;
import :LogStream;

namespace SIByL::Core
{
	export struct Logger
	{
		auto beginLog(char const* type = nullptr) noexcept -> LogStream&;
		auto flushLog() noexcept -> void;

		auto log(std::string const& s) noexcept -> void;
		auto debug(std::string const& s) noexcept -> void;
		auto warning(std::string const& s) noexcept -> void;
		auto error(std::string const& s) noexcept -> void;

		auto clear() noexcept -> void;
		auto pushTimestamp(char const* type = nullptr) noexcept -> void;

		LogStream stream;
	};

	auto Logger::beginLog(char const* type) noexcept -> LogStream&
	{
		clear();
		pushTimestamp(type);
		return stream;
	}

	auto Logger::flushLog() noexcept -> void
	{
		std::cout << stream << std::endl;
	}

	auto Logger::log(std::string const& s) noexcept -> void
	{
		auto& stream = beginLog();
		stream << s.c_str();
		flushLog();
	}

	auto Logger::debug(std::string const& s) noexcept -> void
	{
		auto& stream = beginLog(LogStream::cyan);
		stream << LogStream::cyan << s.c_str() << LogStream::white;
		flushLog();
	}

	auto Logger::warning(std::string const& s) noexcept -> void
	{
		auto& stream = beginLog(LogStream::yellow);
		stream << LogStream::yellow << s.c_str() << LogStream::white;
		flushLog();
	}
	
	auto Logger::error(std::string const& s) noexcept -> void
	{
		auto& stream = beginLog(LogStream::red);
		stream << LogStream::red << s.c_str() << LogStream::white;
		flushLog();
	}

	auto Logger::clear() noexcept -> void
	{
		stream.block.clear();
	}

	auto Logger::pushTimestamp(char const* type) noexcept -> void
	{
		std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
		std::time_t now_time = std::chrono::system_clock::to_time_t(now);
		std::tm tm;
		localtime_s(&tm, &now_time);
		if (type != nullptr) stream << type;
		stream << std::format("[{:02d}:{:02d}:{:02d}] ", tm.tm_hour, tm.tm_min, tm.tm_min).c_str();
	}
}