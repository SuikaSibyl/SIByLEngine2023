module;
#include <limits>
export module Math.Limits;

namespace SIByL::Math
{
	export inline constexpr float float_infinity = std::numeric_limits<float>::infinity();
	export inline constexpr float float_min = std::numeric_limits<float>::min();
	export inline constexpr float float_max = std::numeric_limits<float>::max();
	export inline constexpr float one_minus_epsilon = 0x1.fffffep-1;
}