module;
#include <limits>
#include <cstdint>
export module Math.Limits;

namespace SIByL::Math
{
	export constexpr inline float float_infinity = std::numeric_limits<float>::infinity();
	export constexpr inline float float_min = std::numeric_limits<float>::min();
	export constexpr inline float float_max = std::numeric_limits<float>::max();
	export constexpr inline float one_minus_epsilon = 0x1.fffffep-1;

	export constexpr inline float uint32_max = std::numeric_limits<uint32_t>::max();

	export constexpr inline float shadow_epsilon = 0.0001f;

	export struct Limits {
		static float float_min;
	};

	float Limits::float_min = 0.f;
}