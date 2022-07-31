export module Tracer.Interactions:SurfaceInteraction;
import :Interaction;
import Math.Vector;
import Math.Geometry;
import Tracer.Shape;

namespace SIByL::Tracer
{
	struct Primitive;
	/**
	* Represent local information at a point on a 2D surface.
	* Supply enough information about the surface point.
	*/
	export struct SurfaceInteraction :public Interaction
	{
		SurfaceInteraction();
		SurfaceInteraction(Math::point3 const& p, Math::point3 const& pError,
			Math::point2 const& uv, Math::vec3 const& wo,
			Math::vec3 const& dpdu, Math::vec3 const& dpdv,
			Math::normal3 const& dndu, Math::vec3 const& dndv,
			float time, Shape const* shape);

		/** parameterizeation of the surface */
		Math::point2 uv;

		/** the Primitive that the ray hits */
		Primitive const* primitive = nullptr;

		struct {
			Math::normal3 n;
			Math::vec3 dpdu, dpdv;
			Math::normal3 dndu, dndv;
		} shading;
	};
}