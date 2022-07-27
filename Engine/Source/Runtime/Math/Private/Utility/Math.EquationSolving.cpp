module;
#include <cmath>
#include <algorithm>
module Math.EquationSolving;
import Math.RoundingError;

namespace SIByL::Math
{
	inline auto quadratic(float a, float b, float c, float& t0, float& t1) noexcept -> bool {
		// Find quadratic discriminant
		double discrim = (double)b * (double)b - 4 * (double)a * (double)c;
		if (discrim < 0) return false;
		double rootDiscrim = std::sqrt(discrim);
		// Compute quadratic t value
		double q;
		if (b < 0) q = -.5 * (b - rootDiscrim);
		else       q = -.5 * (b + rootDiscrim);
		t0 = q / a;
		t1 = c / q;
		if (t0 > t1) std::swap(t0, t1);
		return true;
	}
	
	inline auto quadratic(efloat a, efloat b, efloat c, efloat& t0, efloat& t1) noexcept -> bool {
		// Find quadratic discriminant
		double discrim = (double)b * (double)b - 4 * (double)a * (double)c;
		if (discrim < 0) return false;
		double rootDiscrim = std::sqrt(discrim);

		efloat floatRootDiscrim(rootDiscrim, MachineEpsilon * rootDiscrim);

		// Compute quadratic t value
		efloat q;
		if ((float)b < 0) q = efloat(-.5f) * (b - floatRootDiscrim);
		else			  q = efloat(-.5f) * (b + floatRootDiscrim);
		t0 = q / a;
		t1 = c / q;
		if ((float)t0 > (float)t1) std::swap(t0, t1);
		return true;
	}

}