export module Tracer.BSDF;
import Tracer.Interactions;
import Math.Vector;
import Math.Geometry;

namespace SIByL::Tracer
{
	/**
	* BSDF struct represents a collection of BRDFs and BTDFs.
	* BSDF implementation also deal with shading normal problem.
	*/
	export struct BSDF
	{
		/**
		* Compute an orthonormal coordinate system with the shading normal as one of the axes.
		* 
		* @param si: information about the differential geometry at the point on a surface
		* @param eta: relative index of refraction over the boundary
		*/
		BSDF(SurfaceInteraction const& si, float eta = 1);

		float const eta;
		Math::normal3 ns, ng;
		Math::vec3 ss, ts;
	};
}