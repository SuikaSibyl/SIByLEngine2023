module;
export module Tracer.Shape:Shape;
import Math.Transform;
import Math.Geometry;

namespace SIByL::Tracer
{
	struct SurfaceInteraction;

	export struct Shape
	{
		Shape(Math::Transform const* objectToWorld, Math::Transform const* worldToObject, bool reverseOrientation)
			: objectToWorld(objectToWorld)
			, worldToObject(worldToObject)
			, reverseOrientation(reverseOrientation)
			, transformSwapHandedness(objectToWorld->swapsHandness())
		{}

		// 
		virtual auto objectBound() const noexcept -> Math::bounds3 = 0;

		// compute the surface area of a shape in object space
		virtual auto area() const noexcept -> float = 0;

		virtual auto intersect(
			Math::ray3 const& ray, 
			float* tHit,
			SurfaceInteraction* isect,
			bool testAlphaTexture = true) const -> bool = 0;

		virtual auto intersectP(
			Math::ray3 const& ray,
			bool testAlphaTexture = true) const  -> bool;

		auto worldBound() const noexcept -> Math::bounds3;

		// All shapes are defined in object coordinate,
		// use object-to-world Transform to present Transformation
		Math::Transform const* objectToWorld;
		Math::Transform const* worldToObject;
		// Whether surface normal directions should be reversed from default
		bool const reverseOrientation;
		// The value of Transform::SwapsHandedness() of object-to-world Transform
		bool const transformSwapHandedness;
	};
}