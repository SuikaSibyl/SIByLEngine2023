module;
#include <cstdint>
export module Math.Geometry:Ray3;
import Math.Vector;
import Math.Limits;
import :Point3;
import :Normal3;

namespace SIByL::Math
{
	export struct ray3
	{
		ray3() :tMax(Math::float_infinity) {}
		ray3(Math::point3 const& o, Math::vec3 const& d, float tMax = Math::float_infinity)
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