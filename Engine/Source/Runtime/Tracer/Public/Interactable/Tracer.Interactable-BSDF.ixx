export module Tracer.Interactable:BSDF;
import :Interaction.SurfaceInteraction;
import Math.Vector;
import Math.Geometry;
import Tracer.BxDF;
import Tracer.Spectrum;

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

		auto f(Math::vec3 const& woW, Math::vec3 const& wiW, BxDF::Type flags = BxDF::Type::BSDF_ALL) const noexcept -> Spectrum;

		auto sample_f(Math::vec3 const& woWorld, Math::vec3* wiWorld, Math::point2 const& u, float* pdf, BxDF::Type type, BxDF::Type* sampledType = nullptr) const noexcept -> Spectrum;

		float const eta;
		Math::normal3 ns, ng;
		Math::vec3 ss, ts;

		static constexpr int maxBxDFs = 8;
		int nBxDFs = 0;
		BxDF* bxdfs[maxBxDFs];
	};
}