module;
module Tracer.Shape:Shape;
import Tracer.Shape;

import Math.Transform;
import Math.Geometry;
import Tracer.Interactions;

namespace SIByL::Tracer
{
	auto Shape::worldBound() const noexcept -> Math::bounds3 {
		return (*objectToWorld) * (objectBound());
	}

	auto Shape::intersectP(Math::ray3 const& ray, bool testAlphaTexture) const  -> bool {
		float tHit = ray.tMax;
		SurfaceInteraction isect;
		return intersect(ray, &tHit, &isect, testAlphaTexture);
	}
}