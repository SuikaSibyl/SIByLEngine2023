#define DLIB_EXPORT
#include <se.core.hpp>
#undef DLIB_EXPORT
#include <format>
#include <fstream>

namespace se {
timer::timer() {
  startTimePoint = std::chrono::steady_clock::now();
  prevTimePoint = startTimePoint;
}

auto timer::update() noexcept -> void {
  auto const now = std::chrono::steady_clock::now();
  uint64_t const deltaTimeCount = uint64_t(
      std::chrono::duration<double, std::micro>(now - prevTimePoint).count());
  _deltaTime = 0.000001 * deltaTimeCount;
  prevTimePoint = now;
}

auto timer::deltaTime() noexcept -> double { return _deltaTime; }

auto timer::totalTime() noexcept -> double {
  uint64_t const totalTimeCount = uint64_t(
    std::chrono::duration<double, std::micro>(prevTimePoint - startTimePoint).count());
  return 0.000001 * totalTimeCount;
}

auto worldtime::get() noexcept -> worldtime {
  worldtime wtp;
  using namespace std;
  using namespace std::chrono;
  typedef duration<int, ratio_multiply<hours::period, ratio<24> >::type> days;
  system_clock::time_point now = system_clock::now();
  system_clock::duration tp = now.time_since_epoch();
  wtp.y = duration_cast<years>(tp);
  tp -= wtp.y;
  wtp.d = duration_cast<days>(tp);
  tp -= wtp.d;
  wtp.h = duration_cast<hours>(tp);
  tp -= wtp.h;
  wtp.m = duration_cast<minutes>(tp);
  tp -= wtp.m;
  wtp.s = duration_cast<seconds>(tp);
  tp -= wtp.s;
  return wtp;
}

auto worldtime::to_string() noexcept -> std::string {
  std::string str;
  str += std::to_string(y.count());
  str += std::to_string(d.count());
  str += std::to_string(h.count());
  str += std::to_string(m.count());
  str += std::to_string(s.count());
  return str;
}
}