module;
#include <limits>
module Tracer.Ray:Ray;
import Tracer.Ray;
import Math.Vector;
import Math.Geometry;
import Math.Limits;
import Math.Transform;
import Tracer.Medium;

namespace SIByL::Math
{
	inline auto operator*(Transform const& t, Tracer::Ray const& r)->Tracer::Ray {
		Math::ray3 ray = (Math::ray3)r;
		ray = t * ray;
		Tracer::Ray ret;
		ret.o = ray.o;
		ret.d = ray.d;
		ret.tMax = ray.tMax;
		ret.medium = r.medium;
		ret.time = r.time;
		return r;
	}

	inline auto operator*(AnimatedTransform const& t, Tracer::Ray const& r)->Tracer::Ray {
		Math::ray3 ray = (Math::ray3)r;
		Transform interpTrans;
		t.interpolate(r.time, &interpTrans);
		return interpTrans * r;
	}
}