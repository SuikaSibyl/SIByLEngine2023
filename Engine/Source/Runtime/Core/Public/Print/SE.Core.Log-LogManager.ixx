module;
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <functional>
export module SE.Core.Log:LogManager;
import SE.Core.Memory;
import SE.Core.System;

import :LogStream;
import :Logger;

namespace SIByL::Core
{
	/** Log manager is a singleton that helps with logging */
	export struct LogManager :public Manager {
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
		/** a callback func for editor to get logged info */
		std::function<void(std::string const&)> editorCallback = nullptr;
	private:
		/** the global logger */
		Logger logger;
		/** the singleton of LogManager */
		static LogManager* singleton;
	};

#pragma region LOG_MANAGER_IMPL

	LogManager* LogManager::singleton = nullptr;

	auto LogManager::startUp() noexcept -> void {
#ifdef _NEED_LOG
		// duplicate initialize
		if (singleton != nullptr) __debugbreak();
		singleton = this;
#endif
	}

	auto LogManager::shutDown() noexcept -> void {
		singleton = nullptr;
	}

	inline auto LogManager::Debug(std::string const& s) noexcept -> void {
#ifdef _NEED_LOG 
		getLogger().debug(s);
		if (singleton->editorCallback)
			singleton->editorCallback("[D]" + s);
#endif 
	}

	inline auto LogManager::Log(std::string const& s) noexcept -> void {
#ifdef _NEED_LOG 
		getLogger().log(s);
		if (singleton->editorCallback)
			singleton->editorCallback("[L]" + s);
#endif 
	}

	inline auto LogManager::Warning(std::string const& s) noexcept -> void {
#ifdef _NEED_LOG 
		getLogger().warning(s);
		if (singleton->editorCallback)
			singleton->editorCallback("[W]" + s);
#endif 
	}

	inline auto LogManager::Error(std::string const& s) noexcept -> void {
#ifdef _NEED_LOG 
		getLogger().error(s);
		if (singleton->editorCallback)
			singleton->editorCallback("[E]" + s);
#endif 
	}

#pragma endregion
}