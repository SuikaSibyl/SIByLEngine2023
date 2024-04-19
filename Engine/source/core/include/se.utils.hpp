#ifndef SIByL_CORE_MODULE_UTILS
#define SIByL_CORE_MODULE_UTILS
#include <string>
#include <cstdint>
#include <unordered_map>
using RUID = std::uint64_t;
#define ENTT_ID_TYPE std::uint64_t;
#include <ex.entt.hpp>
#include <type_traits>
namespace ex = entt;

#ifdef DLIB_EXPORT
#define SIByL_API __declspec(dllexport)
#else
#define SIByL_API __declspec(dllimport)
#endif

namespace se {
template <class T>
struct Singleton {
  // get the singleton instance
  static T* instance();
  // explicitly release singleton resource
  static void release();
 private:
  Singleton() {}
  Singleton(Singleton<T>&) {}
  Singleton(Singleton<T>&&) {}
  ~Singleton() {}
  Singleton<T>& operator=(Singleton<T> const) {}
  Singleton<T>& operator=(Singleton<T>&&) {}
 private:
  static T* pinstance;
};

#define SINGLETON(T, CTOR)		    \
 private:							\
  friend struct Singleton<T>;		\
  T() CTOR							\
 public:

template <class T>
T* Singleton<T>::instance() {
  if (pinstance == nullptr) pinstance = new T();
  return pinstance;
}
template <class T>
void Singleton<T>::release() {
  if (pinstance != nullptr) delete pinstance;
}
template <class T>
T* Singleton<T>::pinstance = nullptr;

SIByL_API struct RuntimeConfig {
    RuntimeConfig();
    static auto set_config_file(std::string const& path) noexcept -> void;
    static auto get() -> RuntimeConfig const*;
    auto string_property(std::string const& name) const noexcept -> std::string;
    auto string_array_property(std::string const& name) const noexcept -> std::vector<std::string> const&;
    std::unordered_map<std::string, std::string> string_properties;
    std::unordered_map<std::string, std::vector<std::string>> string_array_properties;
    static RuntimeConfig* singleton;
    static std::string config_file_path;
};
}

template <class T>
requires std::is_enum_v<T>
constexpr inline auto hasBit(uint32_t flag, T bit) -> bool {
  return (uint32_t(flag) & static_cast<std::underlying_type<T>::type>(bit)) != 0;
}

template <class T>
requires std::is_enum_v<T>
constexpr inline auto operator|(T lhs, T rhs) -> T {
  return static_cast<T>(static_cast<std::underlying_type<T>::type>(lhs) |
                        static_cast<std::underlying_type<T>::type>(rhs));
}

#endif