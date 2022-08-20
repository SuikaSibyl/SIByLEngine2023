export module Tracer.Interactable:BSSRDF;
import Math.Vector;
import Tracer.Interactable;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	/**
	* BSSRDF
	*/
	export struct BSSRDF
	{
		/**
		* @param po is the current outgoing surface interaction
		* @param eta is the index of refraction of the scattering medium, which is assuemd to be a constant
		*/
		BSSRDF(SurfaceInteraction const& po, float eta) :po(po), eta(eta) {}

		/**
		* Evaluate the eight-dimensional distribution function,
		* which qualifies the ratio of differential radiance at point p_o in direction ¦Ø_o
		* to the incident differential flux at p_i from direction ¦Ø_i.
		*/
		virtual auto s(SurfaceInteraction const& pi, Math::vec3 const& wi) noexcept -> Spectrum;

		SurfaceInteraction const& po;
		float eta;
	};
}