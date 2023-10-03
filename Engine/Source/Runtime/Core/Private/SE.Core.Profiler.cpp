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

}