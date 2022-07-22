export module Tracer.Interactions:SurfaceInteraction;
import :Interaction;
import Math.Vector;
import Math.Geometry;

namespace SIByL::Tracer
{
	struct Primitive;
	/**
	* Represent local information at a point on a 2D surface.
	* Supply enough information about the surface point.
	*/
	export struct SurfaceInteraction :public Interaction
	{
		/** parameterizeation of the surface */
		Math::point2 uv;

		/** the Primitive that the ray hits */
		Primitive const* primitive = nullptr;
	};
}