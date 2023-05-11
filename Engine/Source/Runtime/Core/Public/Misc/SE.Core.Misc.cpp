#include <Misc/SE.Core.Misc.hpp>

namespace SIByL::Core {
Timer::Timer() {
  startTimePoint = std::chrono::steady_clock::now();
  prevTimePoint = startTimePoint;
}
}