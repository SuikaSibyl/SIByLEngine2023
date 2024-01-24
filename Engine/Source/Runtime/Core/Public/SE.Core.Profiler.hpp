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
  ProfileSession(std::string const& name = "Unknown Session");
  ~ProfileSession();
  auto writeHeader() noexcept -> void;
  auto writeFooter() noexcept -> void;
  auto writeSegment(ProfileSegment const& seg) noexcept -> void;
  std::ofstream outputStream;
  uint64_t profileCount = 0;
  InstrumentationSession* currentSession;
  static ProfileSession& get();
  void beginSession(const std::string& name, const std::string& filepath = "profile.json");
  void endSession();
};

SE_EXPORT struct InstrumentationTimer {
 public:
  InstrumentationTimer(const char* name);
  ~InstrumentationTimer();
  void stop();
 private:
  const char* m_Name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;
  bool m_Stopped;
};

#define PROFILE_SCOPE(name) SIByL::Core::InstrumentationTimer timer##__LINE__(name)
#define PROFILE_BEGIN_SESSION(name, filepath) \
 SIByL::Core::ProfileSession::get().beginSession(name, filepath);
#define PROFILE_END_SESSION() SIByL::Core::ProfileSession::get().endSession();
#define PROFILE_SCOPE_FUNCTION() PROFILE_SCOPE(__FUNCSIG__)
}