module;
#include <cstdint>
#include <cmath>
export module Tracer.BxDF:BxDF;
import SE.Math.Misc;
import SE.Math.Geometric;
import Tracer.Spectrum;
import Tracer.Sampling;

namespace SIByL::Tracer
{
	export enum struct TransportMode {
		Radiance,
		Importance,
	};

	/**
	* BxDF is the interface for the individual BRDF and BTDF functions.
	*/
	export struct BxDF
	{
		enum Type {
			BSDF_REFLECTION		= 1 << 0,
			BSDF_TRANSMISSION	= 1 << 1,
			BSDF_DIFFUSE		= 1 << 2,
			BSDF_GLOSSY			= 1 << 3,
			BSDF_SPECULAR		= 1 << 4,
			BSDF_ALL			= BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | 
								  BSDF_REFLECTION | BSDF_TRANSMISSION
		};

		BxDF(Type type) :type(type) {}

		/** determines if the BxDF matches the user-supplied type flags */
		auto matchFlags(Type t) const noexcept -> bool;

		/** return the value of the distribution function for the given pair of directions. */
		virtual auto f(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> Spectrum = 0;

		// -------------------------------
		// Aggregate Behavior of the 4D BxDF
		// -------------------------------
		/**
		* Computes hemispherical-directional reflectance ρ_hd. 
		* Gives the total reflection in a given direction due to constant inllumination over the hemisphere. 
		* Or equivalently total reflection over the hemisphere due to light from a given direction.
		* @param nSamples	: the number of Monte Carlo samples to take to approximate ρ_hd
		* @param samples	: the samples for using Monte Carlo to approximate ρ_hd
		*/
		virtual auto rho(Math::vec3 const& wo, int nSamples, Math::point2 const* samples) const noexcept -> Spectrum;

		/**
		* Computes hemispherical-hemispherical reflectance ρ_hh.
		* Is the fraction of incident light reflected by a surface when the incident light is the same from all directions.
		* @param nSamples	: the number of Monte Carlo samples to take to approximate ρ_hh
		* @param u1/u2		: the samples for using Monte Carlo to approximate ρ_hh
		*/
		virtual auto rho(int nSamples, Math::point2 const* u1, Math::point2 const* u2) const noexcept -> Spectrum;

		// -------------------------------
		// Sampling Reflection Functions
		// -------------------------------
		/*
		* Compute the direction of incident light given an outgoing direction and 
		* returns the value of BxDF for the pair of directions.
		* Handle scattering that is described by delta distribution as well as for randomly
		* sampling directions from BxDFs that scatter light along multiple directions.
		* @param wo		: the given outgoing light
		* @param wi		: 1. (for specular/delta distribution) return the direction of incident light computed according to wi
		*				  2. (else) choose a direction according to a distribution that is similar to its scattering function
		* @param sample	: In the range [0, 1)^2. Not needed for delta distribution BxDFs (specular).
		*				  They are intended to be used by an inversion method-based sampling algorithm.
		* @param pdf	: The value of p(wi), be measured with respect to solid angle. Not needed for delta distribution BxDFs.
		* @return the value of the BSDF for the chosen direction (wi)
		*/
		virtual auto sample_f(Math::vec3 const& wo, Math::vec3* wi, 
			Math::point2 const& sample, float* _pdf, Type* sampledType = nullptr) const noexcept -> Spectrum {
			// default implementation:
			// Samples the unit hemisphere with a cosine-weighted distribution.
			// which is correct for any BRDF that isnt described by a delta distribution.
			
			// Cosine-sample the hemisphere, flipping the direction if necessary
			*wi = cosineSampleHemisphere(sample);
			if (wo.z < 0) wi->z *= -1;
			*_pdf = pdf(wo, *wi);
			return f(wo, *wi);
		}

		/**
		* return the PDF for a given pair of directions
		* @param wo		: the given outgoing light
		* @param wi		: the given incoming light
		*/
		virtual auto pdf(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> float;

		/** Type of the BxDF */
		Type const type;
	};

	/**
	* Geometry setting utility functions.
	* Using reflection coordinate system, which sets orthonormal basis vectors (s,t,n) to (x,y,z) axes.
	* θ: measured from the given direction to the z axis.
	* Φ: angle formed with the x axis after projection of the direction onto the xy plane.
	*/
	/** get the cosine of the angle that is forms with the normal direction */
	export inline auto cosTheta(Math::vec3 const& w) noexcept -> float { return w.z; }
	/** get the cosine square of the angle that is forms with the normal direction */
	export inline auto cos2Theta(Math::vec3 const& w) noexcept -> float { return w.z * w.z; }
	/** get the cosine abs of the angle that is forms with the normal direction */
	export inline auto absCosTheta(Math::vec3 const& w) noexcept -> float { return std::abs(w.z); }
	/** get the sin square of the angle that is forms with the normal direction */
	export inline auto sin2Theta(Math::vec3 const& w) noexcept -> float { return std::max(0.f, 1.f - cos2Theta(w)); }
	/** get the sin of the angle that is forms with the normal direction */
	export inline auto sinTheta(Math::vec3 const& w) noexcept -> float { return std::sqrt(sin2Theta(w)); }
	/** get the tan of the angle that is forms with the normal direction */
	export inline auto tanTheta(Math::vec3 const& w) noexcept -> float { return sinTheta(w) / cosTheta(w); }
	/** get the tan square of the angle that is forms with the normal direction */
	export inline auto tan2Theta(Math::vec3 const& w) noexcept -> float { return sin2Theta(w) / cos2Theta(w); }
	/** get the cos of the angle that is forms with the s/x direction */
	export inline auto cosPhi(Math::vec3 const& w) noexcept -> float {
		float sintheta = sinTheta(w);
		return (sintheta == 0) ? 1 : Math::clamp(w.x / sintheta, -1.f, 1.f);
	}
	/** get the sin of the angle that is forms with the s/x direction */
	export inline auto sinPhi(Math::vec3 const& w) noexcept -> float {
		float sintheta = sinTheta(w);
		return (sintheta == 0) ? 0 : Math::clamp(w.y / sintheta, -1.f, 1.f);
	}
	/** get the cos square of the angle that is forms with the s/x direction */
	export inline auto cos2Phi(Math::vec3 const& w) noexcept -> float { return cosPhi(w) * cosPhi(w); }
	/** get the sin square of the angle that is forms with the s/x direction */
	export inline auto sin2Phi(Math::vec3 const& w) noexcept -> float { return sinPhi(w) * sinPhi(w); }
	/** get the ΔΦ between two vectors in the shading coordinate system */
	export inline auto cosDPhi(Math::vec3 const& wa, Math::vec3 const& wb) noexcept -> float {
		return Math::clamp((wa.x * wb.x + wa.y * wb.y) /
			std::sqrt((wa.x * wa.x + wa.y * wa.y) * (wb.x * wb.x + wb.y * wb.y)), -1.f, 1.f);
	}
	/** tell whether two vectors are in the same hemisphere ( devided by xy plane) */
	export inline auto sameHemisphere(Math::vec3 const& w, Math::vec3 const& wp) noexcept -> bool {
		return w.z * wp.z > 0;
	}
}