#include <Misc/SE.Core.Misc.hpp>
#include <set>

namespace SIByL::Core {
inline const std::set<char> invalid_char = {'\\', '/', ':', '*', '?',
                                            '"',  '<', '>', '|', '.' };

auto StringHelper::invalidFileFolderName(std::string const& name) noexcept
    -> std::string {
  std::string x = name;
  for (char& c : x)
    if (invalid_char.find(c) != invalid_char.end()) c = '_';
  return x;
}

Timer::Timer() {
  startTimePoint = std::chrono::steady_clock::now();
  prevTimePoint = startTimePoint;
}
}