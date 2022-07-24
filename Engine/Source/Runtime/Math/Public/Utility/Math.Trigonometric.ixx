module;
#include <limits>
export module Math.Trigonometric;

namespace SIByL::Math
{
	export inline float float_Pi = 3.14159265358979323846;
	export inline double double_Pi = 3.14159265358979323846;

	export inline auto radians(float deg) noexcept -> float { return (float_Pi / 180) * deg; }
	export inline auto radians(double deg) noexcept -> double { return (double_Pi / 180) * deg; }

}