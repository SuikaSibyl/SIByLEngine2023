module;
#include <vector>
export module Tracer.Interactable:Scene;
import :Interaction;
import :Interaction.SurfaceInteraction;
import :Primitive; 

import Math.Vector;
import Math.Geometry;
import Math.Transform;
import Tracer.Medium;
import Tracer.Spectrum;
import Tracer.Ray;

namespace SIByL::Tracer
{
	export struct Light;

	export struct Scene
	{
		Scene(Primitive* aggregate, std::vector<Light*> const& lights);

		auto getWorldBound() const noexcept -> Math::bounds3 const&;

		/**
		* Traces the given ray into the scene and returns a bool value
		* indicating whether the ray intersected any of the primitives.
		* If so, fills the provided SurfaceInteraction structure with
		* information about the closet intersection point along the ray.
		*/
		auto intersect(Ray const& ray, SurfaceInteraction* isect) const noexcept -> bool;

		/*
		* Checks for the existence of intersections along the ray but does
		* not return any information about those intersections. It is generally
		* more efficient and regularly used for shadow rays.
		*/
		auto intersectP(Ray const& ray) const noexcept -> bool;

		std::vector<Light*> lights;
		Primitive* aggregate;
		Math::bounds3 worldBound;
	};
}