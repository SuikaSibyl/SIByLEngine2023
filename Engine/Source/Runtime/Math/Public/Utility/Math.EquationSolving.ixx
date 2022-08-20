export module Math.EquationSolving;
import Math.RoundingError;

namespace SIByL::Math
{
	export inline auto quadratic(float a, float b, float c, float& t0, float& t1) noexcept -> bool;
	export inline auto quadratic(efloat a, efloat b, efloat c, efloat& t0, efloat& t1) noexcept -> bool;

	export inline auto solveLinearSystem2x2(float const A[2][2], const float B[2], float* x0, float* x1) noexcept -> bool;
}