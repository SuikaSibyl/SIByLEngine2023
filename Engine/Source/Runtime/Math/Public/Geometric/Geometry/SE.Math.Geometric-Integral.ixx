module;
#include <cmath>
#include <limits>
#include <cstdint>
export module SE.Math.Geometric:Integral;
import :Vector3;
import :Point2;
import SE.Math.Misc;

namespace SIByL::Math
{
	export inline auto sphericalDirection(float sinTheta, float cosTheta, float phi) noexcept -> Math::vec3 {
		return Math::vec3(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
	}

	export inline auto sphericalDirection(float sinTheta, float cosTheta, float phi, 
		Math::vec3 const& x, Math::vec3 const& y, Math::vec3 const& z) noexcept -> Math::vec3 
	{
		return sinTheta * std::cos(phi) * x + sinTheta * std::sin(phi) * y + cosTheta * z;
	}

	export inline auto sphericalTheta(Math::vec3 const& v) noexcept -> float {
		return std::acos(Math::clamp(v.z, -1.f, 1.f));
	}

	export inline auto sphericalPhi(Math::vec3 const& v) noexcept -> float {
		float p = std::atan2(v.y, v.x);
		return (p < 0) ? (p + 2 * Math::float_Pi) : p;
	}
}