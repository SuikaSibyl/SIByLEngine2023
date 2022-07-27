module;
#include <limits>
export module Tracer.Ray:Ray;
import Math.Vector;
import Math.Geometry;
import Math.Limits;
import Math.Transform;
import Tracer.Medium;

namespace SIByL::Tracer
{
	export struct Ray
	{
		Ray() :tMax(Math::float_infinity), time(0.f), medium(nullptr) {}
		Ray(Math::point3 const& o, Math::vec3 const& d, float tMax = Math::float_infinity, float time = 0.f, Medium const* medium = nullptr)
			:o(o), d(d), tMax(tMax), time(time), medium(medium) {}

		/** origin */
		Math::point3 o;
		/** direction */
		Math::vec3 d;
		/** restrict the ray to segment [0,r(tMax)]*/
		mutable float tMax;
		/** time, used to handle animated scene*/
		float time;
		/** A medium containing its origin */
		Medium const* medium;

		operator Math::ray3() const { return Math::ray3{ o,d,tMax }; }
		Math::point3 operator()(float t) const { return o + d * t; }
	};
}

namespace SIByL::Math
{
	export inline auto operator*(Transform const& t, Tracer::Ray const& r) -> Tracer::Ray;
	export inline auto operator*(AnimatedTransform const& t, Tracer::Ray const& r) -> Tracer::Ray;
}