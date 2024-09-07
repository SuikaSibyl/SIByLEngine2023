#include <string>
#include <chrono>
#include <thread>
#include "../Public/SE.Core.Profiler.hpp"
#include <Config/SE.Core.Config.hpp>

namespace SIByL::Core {
ProfileSession::ProfileSession(std::string const& name) {
  outputStream.open("./profile/" + name + ".json");
  profileCount = 0;
  writeHeader();
}

ProfileSession::~ProfileSession() {
  writeFooter();
  outputStream.close();
  profileCount = 0;
}

auto ProfileSession::writeHeader() noexcept -> void {
  outputStream << "{\"otherData\": {},\"traceEvents\":[";
  outputStream.flush();
}

auto ProfileSession::writeFooter() noexcept -> void {
  outputStream << "]}";
  outputStream.flush();
}

auto ProfileSession::writeSegment(ProfileSegment const& seg) noexcept -> void {
  if (!outputStream.is_open()) return;
  if (profileCount++ > 0) outputStream << ",";
  std::string name = seg.tag;
  outputStream << "{";
  outputStream << "\"cat\":\"function\",";
  outputStream << "\"dur\":" << (seg.end - seg.start) << ',';
  outputStream << "\"name\":\"" << name << "\",";
  outputStream << "\"ph\":\"X\",";
  outputStream << "\"pid\":0,";
  outputStream << "\"tid\":" << seg.threadID << ",";
  outputStream << "\"ts\":" << seg.start;
  outputStream << "}";
  outputStream.flush();
}

void ProfileSession::beginSession(const std::string& name, const std::string& filepath) {
  outputStream.open(filepath);
  writeHeader();
  currentSession = new InstrumentationSession(name);
}

void ProfileSession::endSession() {
  writeFooter();
  outputStream.close();
  delete currentSession;
  currentSession = nullptr;
  profileCount = 0;
}

 ProfileSession& ProfileSession::get() {
  static ProfileSession instance;
  return instance;
}

InstrumentationTimer::InstrumentationTimer(const char* name)
    : m_Name(name), m_Stopped(false) {
  m_StartTimepoint = std::chrono::high_resolution_clock::now();
}

InstrumentationTimer::~InstrumentationTimer() {
  if (!m_Stopped) stop();
}

void InstrumentationTimer::stop() {
  auto endTimepoint = std::chrono::high_resolution_clock::now();
  uint64_t start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimepoint)
    .time_since_epoch().count();
  uint64_t end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint)
    .time_since_epoch().count();
  uint32_t threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());
  ProfileSession::get().writeSegment({std::string(m_Name), threadID, start, end});
  m_Stopped = true;
}
}