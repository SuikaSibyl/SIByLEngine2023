module Tracer.Interactable:Primitive.GeometricPrimitive;
import Tracer.Interactable;
import Math.Geometry;
import Tracer.Ray;
import Tracer.Shape;
import Tracer.Material;
import Tracer.Medium;

namespace SIByL::Tracer
{
	auto GeometricPrimitive::worldBound() const noexcept -> Math::bounds3 {
		return shape->worldBound();
	}

	auto GeometricPrimitive::intersect(Ray const& r, SurfaceInteraction* isect) const noexcept -> bool {
		float tHit;
		if (!shape->intersect(r, &tHit, isect))
			return false;
		r.tMax = tHit;
		isect->primitive = this;
		// initialize SurfaceInteraction::mediumInterface after Shape intersection
		return true;
	}

	auto GeometricPrimitive::intersectP(Ray const& r) const noexcept -> bool {
		if (!shape->intersectP(r))
			return false;
		// initialize SurfaceInteraction::mediumInterface after Shape intersection
		return true;
	}
}