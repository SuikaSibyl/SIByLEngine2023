#pragma once
#include <common_config.hpp>
#include <string>
#include <fstream>

namespace SIByL::Core {
SE_EXPORT struct ProfileSegment {
  std::string tag;
  uint32_t threadID;
  uint64_t start, end;
};

SE_EXPORT struct InstrumentationSession {
  InstrumentationSession(std::string const& n) : name(n) {}
  std::string name;
};

SE_EXPORT struct ProfileSession {
  ProfileSession(std::string const& name);
  ~ProfileSession();
  auto writeHeader() noexcept -> void;
  auto writeFooter() noexcept -> void;
  auto writeSegment(ProfileSegment const& seg) noexcept -> void;
  std::ofstream outputStream;
  uint64_t profileCount = 0;
};
}