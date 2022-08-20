module;
#include <cstdint>
export module Tracer.BxDF:BxDF;
import Math.Vector;
import Math.Geometry;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	/**
	* BxDF is the interface for the individual BRDF and BTDF functions.
	*/
	export struct BxDF
	{
		enum Type {
			BSDF_REFLECTION		= 1 << 0,
			BSDF_TRANSMISSION	= 1 << 1,
			BSDF_DIFFUSE		= 1 << 1,
			BSDF_GLOSSY			= 1 << 1,
			BSDF_SPECULAR		= 1 << 1,
			BSDF_ALL			= BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | 
								  BSDF_REFLECTION | BSDF_TRANSMISSION
		};

		BxDF(Type type) :type(type) {}

		/** determines if the BxDF matches the user-supplied type flags */
		auto matchFlags(Type t) const noexcept -> bool;

		/**
		* @return the value of the distribution function for the given pair of directions.
		*/
		virtual auto f(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> Spectrum = 0;

		/*
		* Compute the direction of incident light given an outgoing direction and returns
		* the value of BxDF for the pair of directions.
		* Handle scattering that is described by delta distribution as well as for randomly
		* sampling directions from BxDFs that scatter light along multiple directions.
		*/
		virtual auto sample_f(Math::vec3 const& wo, Math::vec3* wi,
			Math::ipoint2 const& sample, float* pdf, Type* sampledType = nullptr) const noexcept -> Spectrum;

		/**
		* Computes the reflectance function ¦Ñ_hd.
		*/
		virtual auto rho(Math::vec3 const& wo, int nSamples, Math::point2 const& samples) const noexcept -> Spectrum;

		virtual auto rho(int nSamples, Math::point2 const& samples1, Math::point2 const& samples2) const noexcept -> Spectrum;

		Type const type;
	};
}