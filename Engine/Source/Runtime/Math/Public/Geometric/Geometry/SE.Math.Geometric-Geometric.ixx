module;
#include <cmath>
#include <limits>
#include <cstdint>
export module SE.Math.Geometric:Geometric;
import SE.Math.Misc;
import :Vector3;
import :Point3;
import :Normal3;

namespace SIByL::Math
{
	export inline auto offsetRayOrigin(Math::point3 const& p, Math::vec3 const& pError, 
		Math::normal3 const& n, Math::vec3 const& w) noexcept -> Math::point3 
	{
		float d = dot((vec3)abs(n), pError);
		Math::vec3 offset = d * Math::vec3(n);
		if (dot(w, n) < 0)
			offset = -offset;
		Math::point3 po = p + offset;
		// Round offset point po away from p 
		for (int i = 0; i < 3; ++i) {
			if (offset[i] > 0) po[i] = nextFloatUp(po[i]);
			else if (offset[i] < 0) po[i] = nextFloatDown(po[i]);
		}
		return po;
	}
}