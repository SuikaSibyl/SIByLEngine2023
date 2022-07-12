module;
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
export module Core.Log:LogManager;
import Core.Memory;
import Core.System;

import :LogStream;
import :Logger;

namespace SIByL::Core
{
	export struct LogManager :public Manager
	{
		virtual auto startUp() noexcept -> void override;
		virtual auto shutDown() noexcept -> void override;

		static auto getLogger() noexcept -> Logger& { return get()->logger; }

		static auto Debug(std::string const& s) noexcept -> void;
		static auto Log(std::string const& s) noexcept -> void;
		static auto Error(std::string const& s) noexcept -> void;

	private:
		Logger logger;

		static auto get() noexcept -> LogManager* { return singleton; }
		static LogManager* singleton;
	};

	auto LogManager::startUp() noexcept -> void
	{
#ifdef _NEED_LOG
		// duplicate initialize
		if (singleton != nullptr) __debugbreak();
		singleton = this;
#endif
	}

	auto LogManager::shutDown() noexcept -> void
	{
		singleton = nullptr;
	}

	auto LogManager::Debug(std::string const& s) noexcept -> void { 
#ifdef _NEED_LOG 
		getLogger().debug(s); 
#endif 
	}

	auto LogManager::Log(std::string const& s) noexcept -> void {
#ifdef _NEED_LOG 
		getLogger().log(s);
#endif 
	}

	auto LogManager::Error(std::string const& s) noexcept -> void {
#ifdef _NEED_LOG 
		getLogger().error(s);
#endif 
	}

	LogManager* LogManager::singleton = nullptr;
}