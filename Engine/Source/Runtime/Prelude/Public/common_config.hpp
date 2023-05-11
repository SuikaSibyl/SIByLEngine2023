#pragma once

#ifndef _USE_MODULE_
#define SE_EXPORT  
#endif  // !_USE_MODULE_

#include <string>
#include <type_traits>

namespace SIByL {
SE_EXPORT struct SIByL_Config { static std::string version; };

/** Combine two bitflags. */
SE_EXPORT template <class T>
  requires std::is_enum_v<T>
constexpr inline auto operator|(T lhs, T rhs) -> T {
  return static_cast<T>(static_cast<std::underlying_type<T>::type>(lhs) |
                        static_cast<std::underlying_type<T>::type>(rhs));
}

/** Test whether there is a intersection of two bitflags. */
SE_EXPORT template <class T>
  requires std::is_enum_v<T>
constexpr inline auto operator&(T lhs, T rhs) -> T {
  return static_cast<T>(static_cast<std::underlying_type<T>::type>(lhs) &
                        static_cast<std::underlying_type<T>::type>(rhs));
}

/** Whether flag has the bit. */
SE_EXPORT template <class T>
  requires std::is_enum_v<T>
constexpr inline auto hasBit(T flag, T bit) -> bool {
  return (static_cast<std::underlying_type<T>::type>(flag) &
          static_cast<std::underlying_type<T>::type>(bit)) != 0;
}
}  // namespace SIByL