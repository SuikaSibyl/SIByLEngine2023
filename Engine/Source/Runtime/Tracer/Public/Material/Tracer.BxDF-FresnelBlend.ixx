module;
#include <cmath>
export module Tracer.BxDF:FresnelBlend;
import :BxDF;
import :Fresnel;
import :MicrofacetDistribution;
import Math.Vector;
import Math.Geometry;
import Math.Trigonometric;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	export struct FresnelBlend :public BxDF
	{
		FresnelBlend(Spectrum const& Rd, Spectrum const& Rs, MicrofacetDistribution* distribution)
			: BxDF(BxDF::Type(BSDF_REFLECTION | BSDF_GLOSSY))
			, Rd(Rd), Rs(Rs), distribution(distribution) {}

		auto SchlickFresnel(float cosTheta) const noexcept -> Spectrum {
			auto pow5 = [](float v) { return (v * v) * (v * v) * v; };
			return Rs + pow5(1 - cosTheta) * (Spectrum(1.) - Rs);
		}

		virtual auto f(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> Spectrum override {
			auto pow5 = [](float v) { return (v * v) * (v * v) * v; };
			Spectrum diffuse = (28.f / (23.f * Math::float_Pi)) * Rd * (Spectrum(1.f) - Rs) *
				(1 - pow5(1 - .5f * absCosTheta(wi))) * (1 - pow5(1 - .5f * absCosTheta(wo)));
			Math::vec3 wh = wi + wo;
			if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Spectrum(0);
			wh = Math::normalize(wh);
			Spectrum specular = distribution->D(wh) / (4 * absDot(wi, wh) * 
				std::max(absCosTheta(wi), absCosTheta(wo))) * SchlickFresnel(Math::dot(wi, wh));
			return diffuse + specular;
		}

		auto sample_f(Math::vec3 const& wo, Math::vec3* wi, 
			Math::point2 const& uOrig, float* _pdf, Type* sampledType) const noexcept -> Spectrum
		{
			Math::point2 u = uOrig;
			if (u[0] < .5) {
				u[0] = 2 * u[0];
				// Cosine - sample the hemisphere, flipping the direction if necessary
			}
			else {
				u[0] = 2 * (u[0] - .5f);
				// Sample microfacet orientation ωhand reflected direction ωi
			}
			*_pdf = pdf(wo, *wi);
			return f(wo, *wi);
		}

		virtual auto pdf(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> float override {
			if (!sameHemisphere(wo, wi)) return 0;
			Math::vec3 wh = Math::normalize(wo + wi);
			float pdf_wh = distribution->pdf(wo, wh);
			return .5f * (absCosTheta(wi) * Math::float_InvPi +
				pdf_wh / (4 * Math::dot(wo, wh)));
		}

	private:
		Spectrum const Rd, Rs;
		MicrofacetDistribution* distribution;
	};
}