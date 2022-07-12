module;
#include <limits>
export module Tracer.Ray:Ray;
import Math.Vector;
import Math.Geometry;
import Math.Limits;
import GFX.Medium;

namespace SIByL::Tracer
{
	export struct Ray
	{
		Ray() :tMax(Math::float_infinity), time(0.f), medium(nullptr) {}
		Ray(Math::point3 const& o, Math::vec3 const& d, float tMax = Math::float_infinity, float time = 0.f, GFX::Medium const* medium = nullptr)
			:o(o), d(d), tMax(tMax), time(time), medium(medium) {}

		Math::point3 o;
		Math::vec3 d;

		mutable float tMax;
		float time;
		GFX::Medium const* medium;

		Math::point3 operator()(float t) const { return o + d * t; }
	};
}