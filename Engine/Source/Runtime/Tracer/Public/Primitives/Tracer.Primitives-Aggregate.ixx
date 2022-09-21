module;
#include <vector>
export module Tracer.Primitives:Aggregate;
import Math.Geometry;
import Tracer.Ray;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	/**
	* A contrainer that can hold many Primitives.
	* The base for accleration structures.
	*/
	export struct Aggregate :public Primitive {};

	export struct DummyAggregate :public Aggregate {
		DummyAggregate(std::vector<Primitive*> const& primitives)
			:primitives(primitives)
		{
			for (auto& primitive : primitives)
				worldBounds = Math::unionBounds(worldBounds, primitive->worldBound());
		}

		/** A box that enclose the primitive's geometry in world space */
		virtual auto worldBound() const noexcept -> Math::bounds3 { return worldBounds; }

		/**
		* Update Ray::tMax with distance value if an intersection is found.
		* Initialize additional SurfaceInteraction member variables.
		*/
		virtual auto intersect(Ray const& r, SurfaceInteraction* i) const noexcept -> bool {
			bool hit = false;
			for (auto& primitive : primitives)
				if (primitive->intersect(r, i))
					hit = true;
			return hit;
		}

		virtual auto intersectP(Ray const& r) const noexcept -> bool {
			bool hit = false;
			for (auto& primitive : primitives)
				if (primitive->intersectP(r))
					hit = true;
			return hit;
		}

		/** Describes the primitive's emission distribution if it's a light source */
		virtual auto getAreaLight() const noexcept -> AreaLight const* { return nullptr; }
		/** Return a pointer to the material instance assigned to the primitive */
		virtual auto getMaterial() const noexcept -> Material const* { return nullptr; }

		/**
		* Initialize representations of the light-scattering properties of
		* the material at the intersection point on the surface.
		*/
		virtual auto computeScatteringFunctions(
			SurfaceInteraction* isec,
			Core::MemoryArena& arena,
			TransportMode mode,
			bool allowMultipleLobes) const noexcept -> void override {}

	protected:
		/** world bounds for all primitives */
		Math::bounds3 worldBounds;
		/** all primitives in the aggregate */
		std::vector<Primitive*> primitives;
	};
}