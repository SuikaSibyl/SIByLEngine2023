module;
#include <cmath>
#include <algorithm>
export module SE.Math.Misc:EquationSolving;
import :RoundingError;

namespace SIByL::Math
{
	export inline auto quadratic(float a, float b, float c, float& t0, float& t1) noexcept -> bool;
	export inline auto quadratic(efloat a, efloat b, efloat c, efloat& t0, efloat& t1) noexcept -> bool;

	export inline auto solveLinearSystem2x2(float const A[2][2], const float B[2], float* x0, float* x1) noexcept -> bool;
}

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
  t0 = static_cast<float>(q / a);
  t1 = static_cast<float>(c / q);
  if (t0 > t1) std::swap(t0, t1);
  return true;
}

inline auto quadratic(efloat a, efloat b, efloat c, efloat& t0, efloat& t1) noexcept -> bool {
  // Find quadratic discriminant
  double discrim = (double)b * (double)b - 4 * (double)a * (double)c;
  if (discrim < 0) return false;
  double rootDiscrim = std::sqrt(discrim);

  efloat floatRootDiscrim(static_cast<float>(rootDiscrim), static_cast<float>(MachineEpsilon * rootDiscrim));

  // Compute quadratic t value
  efloat q;
  if ((float)b < 0) q = efloat(-.5f) * (b - floatRootDiscrim);
  else			  q = efloat(-.5f) * (b + floatRootDiscrim);
  t0 = q / a;
  t1 = c / q;
  if ((float)t0 > (float)t1) std::swap(t0, t1);
  return true;
}

inline auto solveLinearSystem2x2(float const A[2][2], const float B[2], float* x0, float* x1) noexcept -> bool {
  float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  if (std::abs(det) < 1e-10f)
    return false;
  *x0 = (A[1][1] * B[0] - A[0][1] * B[1]) / det;
  *x1 = (A[0][0] * B[1] - A[1][0] * B[0]) / det;
  if (std::isnan(*x0) || std::isnan(*x1))
    return false;
  return true;
}
}