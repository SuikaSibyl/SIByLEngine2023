module;
export module Tracer.BxDF:MicrofacetTransmission;
import :BxDF;
import :SpecularTransmission;
import :Fresnel;
import :MicrofacetDistribution;
import Math.Vector;
import Math.Geometry;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	export struct MicrofacetTransmission :public BxDF
	{
		MicrofacetTransmission(const Spectrum& T, MicrofacetDistribution* distribution,
			float etaA, float etaB, TransportMode mode)
			: BxDF(BxDF::Type(BSDF_TRANSMISSION | BSDF_GLOSSY))
			, T(T), distribution(distribution), etaA(etaA), etaB(etaB)
			, fresnel(etaA, etaB), mode(mode) {}

		virtual auto f(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> Spectrum override {
			// TODO
			return Spectrum{};
		}

		virtual auto sample_f(Math::vec3 const& wo, Math::vec3* wi,
			Math::point2 const& u, float* _pdf, Type* sampledType = nullptr) const noexcept -> Spectrum
		{
			Math::vec3 wh = distribution->sample_wh(wo, u);
			float eta = cosTheta(wo) > 0 ? (etaA / etaB) : (etaB / etaA);
			if (!refract(wo, (Math::normal3)wh, eta, wi))
				return 0;
			*_pdf = pdf(wo, *wi);
			return f(wo, *wi);
		}

		virtual auto pdf(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> float override {
			if (sameHemisphere(wo, wi)) return 0;
			// Compute ωh from ωoand ωi for microfacet transmission
			float eta = cosTheta(wo) > 0 ? (etaB / etaA) : (etaA / etaB);
			Math::vec3 wh = Math::normalize(wo + wi * eta);
			// Compute change of variables dwh_dwi for microfacet transmission
			// TODO:
			return distribution->pdf(wo, wh);//TODO: *dwh_dwi;
		}

	private:
		Spectrum const T;
		MicrofacetDistribution const* distribution;
		float const etaA, etaB;
		FresnelDielectric const fresnel;
		TransportMode const mode;
	};
}