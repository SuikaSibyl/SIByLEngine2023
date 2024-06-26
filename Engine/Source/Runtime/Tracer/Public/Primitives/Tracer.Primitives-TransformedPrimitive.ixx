export module Tracer.Primitives:TransformedPrimitive;
import SE.Math.Geometric;
import Tracer.Ray;
import Tracer.Interactable;

namespace SIByL::Tracer
{
	/**
	* Handles two more general uses of Shapes in the scene:
	* Shapes with animated transformation matrices & object instancing.
	*/
	export struct TransformedPrimitive :public Primitive
	{
		TransformedPrimitive(Primitive* primitive, Math::AnimatedTransform const* primitiveToWorld)
			:primitive(primitive), primitiveToWorld(primitiveToWorld) 
		{}

		virtual auto worldBound() const noexcept -> Math::bounds3 override;

		virtual auto intersect(Ray const& r, SurfaceInteraction* i) const noexcept -> bool override;
		virtual auto intersectP(Ray const& r) const noexcept -> bool override;

		Primitive* primitive;
		Math::AnimatedTransform const* primitiveToWorld;
	};
}