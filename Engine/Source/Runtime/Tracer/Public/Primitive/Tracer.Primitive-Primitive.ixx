export module Tracer.Primitive:Primitive;
import Core.Memory;
import Math.Geometry;
import Tracer.Ray;
import Tracer.Interactions;

namespace SIByL::Tracer
{
	struct AreaLight;
	struct Material;
	/*
	* Primitve is the bridge between the geometry processing & shading subsystem of pbrt
	*/
	export struct Primitive
	{
		/** A box that enclose the primitive's geometry in world space */
		virtual auto worldBound() const noexcept -> Math::bounds3 = 0;

		/**
		* Update Ray::tMax with distance value if an intersection is found.
		* Initialize additional SurfaceInteraction member variables.
		*/
		virtual auto intersect(Ray const& r, SurfaceInteraction* i) const noexcept -> bool = 0;
		virtual auto intersectP(Ray const& r) const noexcept -> bool = 0;

		/** Describes the primitive's emission distribution if it's a light source */
		virtual auto getAreaLight() const noexcept -> AreaLight const* = 0;
		/** Return a pointer to the material instance assigned to the primitive */
		virtual auto getMaterial() const noexcept -> Material const* = 0;

		//virtual auto computeScatteringFunctions(
		//	SurfaceInteraction* isec,
		//	Core::MemoryArena& arena,

		//	) const noexcept ->
	};
}