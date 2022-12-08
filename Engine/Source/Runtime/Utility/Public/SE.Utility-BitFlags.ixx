module;
#include <type_traits>
export module SE.Utility:BitFlags;

namespace SIByL
{
    /** Combine two bitflags. */
    export
    template<class T>
    requires std::is_enum_v<T>
    constexpr inline auto operator|(T lhs, T rhs) -> T {
        return static_cast<T>(
            static_cast<std::underlying_type<T>::type>(lhs) |
            static_cast<std::underlying_type<T>::type>(rhs));
    }

    /** Test whether there is a intersection of two bitflags. */
    export
    template<class T>
    requires std::is_enum_v<T>
    constexpr inline auto operator&(T lhs, T rhs) -> T {
        return static_cast<T>(
            static_cast<std::underlying_type<T>::type>(lhs) &
            static_cast<std::underlying_type<T>::type>(rhs));
    }

    /** Whether flag has the bit. */
    export
    template<class T>
    requires std::is_enum_v<T>
    constexpr inline auto hasBit(T flag, T bit) -> bool {
        return (static_cast<std::underlying_type<T>::type>(flag) &
            static_cast<std::underlying_type<T>::type>(bit)) != 0;
    }
}
