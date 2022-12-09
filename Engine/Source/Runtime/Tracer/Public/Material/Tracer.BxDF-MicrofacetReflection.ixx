module;
export module Tracer.BxDF:MicrofacetReflection;
import :BxDF;
import :SpecularReflection;
import :Fresnel;
import :MicrofacetDistribution;
import SE.Math.Geometric;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	export struct MicrofacetReflection :public BxDF
	{
		MicrofacetReflection(const Spectrum& R,
			MicrofacetDistribution* distribution, Fresnel* fresnel)
			: BxDF(BxDF::Type(BSDF_REFLECTION | BSDF_GLOSSY))
			, R(R), distribution(distribution), fresnel(fresnel) {}

		virtual auto f(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> Spectrum override {
			float cosThetaO = absCosTheta(wo), cosThetaI = absCosTheta(wi);
			Math::vec3 wh = wi + wo;
			// Handle degenerate cases for microfacet reflection
			if (cosThetaI == 0 || cosThetaO == 0) return Spectrum(0.);
			if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Spectrum(0.);
			wh = Math::normalize(wh);
			Spectrum F = fresnel->evaluate(Math::dot(wi, wh));
			return R * distribution->D(wh) * distribution->G(wo, wi) * F / (4 * cosThetaI * cosThetaO);
		}

		virtual auto sample_f(Math::vec3 const& wo, Math::vec3* wi,
			Math::point2 const& u, float* pdf, Type* sampledType = nullptr) const noexcept -> Spectrum 
		{
			// Sample microfacet orientation ωhand reflected direction ωi
			Math::vec3 wh = distribution->sample_wh(wo, u);
			*wi = reflect(wo, wh);
			if (!sameHemisphere(wo, *wi)) return Spectrum(0.f);
			// Compute PDF of wi for microfacet reflection 813
			return f(wo, *wi);
		}

		virtual auto pdf(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> float override {
			if (!sameHemisphere(wo, wi)) return 0;
			Math::vec3 wh = Math::normalize(wo + wi);
			return distribution->pdf(wo, wh) / (4 * Math::dot(wo, wh));
		}

	private:
		Spectrum const R;
		MicrofacetDistribution const* distribution;
		Fresnel const* fresnel;
	};
}