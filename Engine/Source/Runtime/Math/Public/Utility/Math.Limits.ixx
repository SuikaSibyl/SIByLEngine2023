module;
#include <limits>
export module Math.Limits;

namespace SIByL::Math
{
	export inline float float_infinity = std::numeric_limits<float>::infinity();
	export inline float float_min = std::numeric_limits<float>::min();
	export inline float float_max = std::numeric_limits<float>::max();
}