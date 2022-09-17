module;
#include <cmath>
export module Tracer.BxDF:SpecularTransmission;
import :BxDF;
import :Fresnel;
import Math.Vector;
import Math.Geometry;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	/** Compute the refracted direction w_t */
	export inline auto refract(Math::vec3 const& wi, Math::normal3 const& n, float eta, Math::vec3* wt) noexcept -> bool {
		// compute cos θ_t using Snell’s law
		float cosThetaI = Math::dot(n, wi);
		float sin2ThetaI = std::max(0.f, 1.f - cosThetaI * cosThetaI);
		float sin2ThetaT = eta * eta * sin2ThetaI;
		//  Handle total internal reflection for transmission
		if (sin2ThetaT >= 1) return false;
		float cosThetaT = std::sqrt(1 - sin2ThetaT);
		*wt = eta * -wi + (eta * cosThetaI - cosThetaT) * Math::vec3(n);
		return true;
	}

	/** The BTDF for specular transmission. */
	export struct SpecularTransmission :public BxDF
	{		
		/**
		* @param R		: the spectrum to scale the reflected color
		* @param etaA	: the index of refraction above the surface
		* @param etaB	: the index of refraction below the surface
		* @param TransportMode: indicates whether the incident ray that intersected the point where the BxDF was 
						  computed started from a light source or whether it was started from the camera.
		*/
		SpecularTransmission(Spectrum const& T, float etaA, float etaB, TransportMode mode)
			: BxDF(BxDF::Type(BSDF_REFLECTION | BSDF_SPECULAR))
			, T(T), etaA(etaA), etaB(etaB), fresnel(etaA, etaB), mode(mode) {}

		/** For an arbitrary pair of directions the delta function returns no scattering, since it's a delta distribution */
		virtual auto f(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> Spectrum override {
			return Spectrum(0.f);
		}

		virtual auto sample_f(Math::vec3 const& wo, Math::vec3* wi,
			Math::point2 const& sample, float* _pdf, Type* sampledType = nullptr) const noexcept -> Spectrum override {
			// figure out which η is incident and which is transmitted
			bool entering = cosTheta(wo) > 0;
			float etaI = entering ? etaA : etaB;
			float etaT = entering ? etaB : etaA;
			// compute ray direction for specular transmission
			if (!refract(wo, Math::faceforward(Math::normal3(0, 0, 1), wo), etaI / etaT, wi))
				return 0;
			*_pdf = 1.f;
			Spectrum ft = T * (Spectrum(1.f) - fresnel.evaluate(cosTheta(*wi)));
			// account for non-symmetry with transmission to different medium
			return ft / absCosTheta(*wi);
		}

		virtual auto pdf(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> float override {
			return 0;
		}

	private:
		Spectrum const T;
		float const etaA, etaB;
		/** conductors do not transmit light, so always use FresnelDielectric */
		FresnelDielectric const fresnel;
		TransportMode const mode;
	};
}