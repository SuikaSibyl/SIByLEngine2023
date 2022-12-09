module;
#include <cstdint>
export module SE.Math.Geometric:Ray3;
import :Vector3;
import :Point3;
import :Normal3;
import SE.Math.Misc;

namespace SIByL::Math
{
	export struct ray3
	{
		ray3() :tMax(float_infinity) {}
		ray3(Math::point3 const& o, Math::vec3 const& d, float tMax = float_infinity)
			:o(o), d(d), tMax(tMax) {}

		auto operator()(float t) const -> Math::point3;

		/** origin */
		point3 o;
		/** direction */
		vec3 d;
		/** restrict the ray to segment [0,r(tMax)]*/
		mutable float tMax;
	};

	auto ray3::operator()(float t) const->Math::point3 {
		return o + d * t;
	}

}