module Tracer.Primitives:GeometricPrimitive;
import Tracer.Primitives;
import Tracer.Interactable;
import SE.Math.Geometric;
import Tracer.Ray;
import Tracer.Shape;
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

	auto GeometricPrimitive::getAreaLight() const noexcept -> AreaLight const* {
		return areaLight;
	}
	
	auto GeometricPrimitive::computeScatteringFunctions(
		SurfaceInteraction* isect,
		Core::MemoryArena& arena,
		TransportMode mode,
		bool allowMultipleLobes) const noexcept -> void
	{
		if (material)
			material->computeScatteringFunctions(isect, arena, mode, allowMultipleLobes);
	}

}