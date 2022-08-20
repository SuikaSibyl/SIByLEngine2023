module;
export module Tracer.Shape:Shape;
import Math.Transform;
import Math.Geometry;

namespace SIByL::Tracer
{
	export struct SurfaceInteraction;
	
	/**
	* A general Shape interface.
	*/
	export struct Shape
	{
		Shape(Math::Transform const* objectToWorld, Math::Transform const* worldToObject, bool reverseOrientation)
			: objectToWorld(objectToWorld)
			, worldToObject(worldToObject)
			, reverseOrientation(reverseOrientation)
			, transformSwapsHandedness(objectToWorld->swapsHandness())
		{}

		/** return a bounding box in shape's object space */
		virtual auto objectBound() const noexcept -> Math::bounds3 = 0;
		/** return a bounding box in world space */
		auto worldBound() const noexcept -> Math::bounds3;

		/**
		* Returns geometric information about a single ray–shape intersection,
		* corresponding to the first intersection, if any, in the (0, tMax)
		* parametric range along the ray;
		* @param ray: input ray is in world space;
		* @param tHit: the parametric distance along the ray;
		* @param isect: a capture of local geometric properties of a surface, in world space;
		* @param testAlphaTexture: indicate whether perfirm a texture base surface cutting;
		*/
		virtual auto intersect(
			Math::ray3 const& ray, 
			float* tHit,
			SurfaceInteraction* isect,
			bool testAlphaTexture = true) const -> bool = 0;

		/**
		* A predicate version of intersect(), which only determines whether or not
		* an intersection occurs, without returning any details about the intersection.
		* The default impl directly call intersect, which is wastefull.
		* @see intersect()
		*/
		virtual auto intersectP(
			Math::ray3 const& ray,
			bool testAlphaTexture = true) const  -> bool;

		/**
		* Compute the surface area of a shape in object space.
		* It is necessary when use Shapes as area lights.
		*/
		virtual auto area() const noexcept -> float = 0;

		/**
		* All shapes are defined in object coordinate,
		* use object-to-world Transform to present Transformation
		*/
		Math::Transform const* objectToWorld;
		Math::Transform const* worldToObject;
		/** Whether surface normal directions should be reversed from default */
		bool const reverseOrientation;
		/** The value of Transform::SwapsHandedness() of object-to-world Transform */
		bool const transformSwapsHandedness;
	};
}