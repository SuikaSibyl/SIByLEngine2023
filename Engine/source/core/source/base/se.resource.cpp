#define DLIB_EXPORT
#include <se.core.hpp>
#undef DLIB_EXPORT
#include <random>
#include <thread>

namespace se {
auto root::resource::add_path(std::filesystem::path const& path) noexcept -> void {
}

auto root::resource::find_path(std::filesystem::path const& path) noexcept -> std::filesystem::path {
  return {};
}

struct ResourceManager {
  std::unordered_map<RUID, std::string> map;
};

static ResourceManager resourceManager;

RUID generateRUID() {
  static std::default_random_engine e;
  static std::uniform_int_distribution<uint64_t> u(0, 0X3FFFFF);
  
  RUID id = 0;
  time_t now = time(0);
  tm ltm;
  localtime_s(&ltm, &now);
  id += (uint64_t(ltm.tm_year - 100) & 0xFF) << 56;
  id += (uint64_t(ltm.tm_mon) & 0xF) << 52;
  id += (uint64_t(ltm.tm_mday) & 0x1F) << 47;
  id += (uint64_t(ltm.tm_hour) & 0x1F) << 42;
  id += (uint64_t(ltm.tm_min) & 0x3F) << 36;
  id += (uint64_t(ltm.tm_sec) & 0x3F) << 30;
 
  std::thread::id tid = std::this_thread::get_id();
  unsigned int nId = *(unsigned int*)((char*)&tid);
  id += (uint64_t(nId) & 0xFF) << 22;
  id += u(e);
  return id;
}

auto root::resource::queryRUID(std::string info) noexcept -> RUID {
  RUID guid = generateRUID();
  while (resourceManager.map.find(guid) != resourceManager.map.end()) {
	guid = generateRUID(); }
  resourceManager.map[guid] = info;
  return guid;
}
}