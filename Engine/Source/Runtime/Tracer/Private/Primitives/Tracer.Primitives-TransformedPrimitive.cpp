module Tracer.Primitives:GeometricPrimitive;
import Tracer.Primitives;
import Tracer.Interactable;
import SE.Math.Geometric;
import Tracer.Ray;

namespace SIByL::Tracer
{
	auto TransformedPrimitive::worldBound() const noexcept -> Math::bounds3 {
		return primitiveToWorld->motionBounds(primitive->worldBound());
	}

	auto TransformedPrimitive::intersect(Ray const& r, SurfaceInteraction* isect) const noexcept -> bool {
		// Compute ray after transformation by PrimitiveToWorld
		Math::Transform interpolatedPrimToWorld;
		primitiveToWorld->interpolate(r.time, &interpolatedPrimToWorld);
		Ray ray = Math::inverse(interpolatedPrimToWorld) * r;
		if (!primitive->intersect(ray, isect))
				return false;
		r.tMax = ray.tMax;
		// Transform instance's intersection data to world space
		if (!interpolatedPrimToWorld.isIdentity())
			*isect = interpolatedPrimToWorld * (*isect);
		return true;
	}

	auto TransformedPrimitive::intersectP(Ray const& r) const noexcept -> bool {
		// Compute ray after transformation by PrimitiveToWorld
		Math::Transform interpolatedPrimToWorld;
		primitiveToWorld->interpolate(r.time, &interpolatedPrimToWorld);
		Ray ray = Math::inverse(interpolatedPrimToWorld) * r;
		if (!primitive->intersectP(ray))
			return false;
		r.tMax = ray.tMax;
		return true;
	}

}