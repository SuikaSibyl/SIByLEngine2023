module;
#include <limits>
export module SE.Math.Misc:Trigonometric;

namespace SIByL::Math
{
	export inline float float_Pi = 3.14159265358979323846;
	export inline double double_Pi = 3.14159265358979323846;

	export inline float float_InvPi = 1.f / float_Pi;
	export inline float float_Inv2Pi = 1.f / (2 * float_Pi);
	export inline float float_Inv4Pi = 1.f / (4 * float_Pi);
	export inline float float_PiOver2 = float_Pi / 2;
	export inline float float_PiOver4 = float_Pi / 4;

	export inline float double_InvPi = 1. / double_Pi;
	export inline float double_Inv2Pi = 1. / (2 * double_Pi);
	export inline float double_Inv4Pi = 1. / (4 * double_Pi);
	export inline float double_PiOver2 = double_Pi / 2;
	export inline float double_PiOver4 = double_Pi / 4;

	export inline auto radians(float deg) noexcept -> float { return (float_Pi / 180) * deg; }
	export inline auto radians(double deg) noexcept -> double { return (double_Pi / 180) * deg; }

}