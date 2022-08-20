export module Tracer.Interactable:Primitive.GeometricPrimitive;
import :Primitive;
import :Light.AreaLight;
import Math.Geometry;
import Tracer.Ray;
import Tracer.Shape;
import Tracer.Material;
import Tracer.Medium;

namespace SIByL::Tracer
{
	/**
	* Shapes to be rendered directly.
	* Combines a Shape with a description of its appearance properties (Materail)
	*/
	export struct GeometricPrimitive :public Primitive
	{
		virtual auto worldBound() const noexcept -> Math::bounds3 override;

		virtual auto intersect(Ray const& r, SurfaceInteraction* i) const noexcept -> bool override;
		virtual auto intersectP(Ray const& r) const noexcept -> bool override;

		Shape*		shape;
		Material*	material;
		AreaLight*	areaLight;
		MediumInterface mediumInterface;
	};
}