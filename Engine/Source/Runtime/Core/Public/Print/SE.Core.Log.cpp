#include <cstdint>
#include <memory>
#include <ostream>

#include "SE.Core.Log.hpp"

namespace SIByL::Core {
#pragma region LOG_STREAM_BLOCK_IMPL

LogStreamBlock::LogStreamBlock() {
  size = 0;
  capacity = 8192;
  data = new char8_t[8192];
}

LogStreamBlock::~LogStreamBlock() { delete[] data; }
#pragma endregion
}  // namespace SIByL::Core

namespace SIByL::Core {
#pragma region LOG_MANAGER_IMPL

LogManager* LogManager::singleton = nullptr;

auto LogManager::startUp() noexcept -> void {
#ifdef _NEED_LOG
  // duplicate initialize
  if (singleton != nullptr) __debugbreak();
  singleton = this;
#endif
}

auto LogManager::shutDown() noexcept -> void { singleton = nullptr; }
#pragma endregion
}