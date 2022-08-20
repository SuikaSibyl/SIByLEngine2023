export module Tracer.Interactable:Primitive.GeometricPrimitive;
import :Primitive;
import :Light.AreaLight;
import Math.Geometry;
import Tracer.Ray;
import Tracer.Shape;
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

		virtual auto getAreaLight() const noexcept -> AreaLight const* override;

		virtual auto computeScatteringFunctions(
			SurfaceInteraction* isec,
			Core::MemoryArena& arena,
			TransportMode mode,
			bool allowMultipleLobes) const noexcept -> void override;

		Shape*		shape;
		Material*	material;
		AreaLight*	areaLight;
		MediumInterface mediumInterface;
	};
}