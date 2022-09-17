module;
#include <cstdint>
module Tracer.BxDF:BxDF;
import Tracer.BxDF;
import Math.Vector;
import Math.Common;
import Math.Geometry;
import Math.Trigonometric;
import Tracer.Spectrum;
import Tracer.Sampling;

namespace SIByL::Tracer
{
	auto BxDF::matchFlags(Type t) const noexcept -> bool {
		return ((uint32_t)type & (uint32_t)t) == (uint32_t)type;
	}

	auto BxDF::sample_f(Math::vec3 const& wo, Math::vec3* wi, Math::point2 const& u, float* _pdf, Type* sampledType) const noexcept -> Spectrum {
		// Cosine - sample the hemisphere, flipping the direction if necessary
		*wi = cosineSampleHemisphere(u);
		if (wo.z < 0) wi->z *= -1;
		*_pdf = pdf(wo, *wi);
		return f(wo, *wi);
	}

	auto BxDF::pdf(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> float {
		// Defautly evaluate the PDF for the cosine-weighted sampling method.
		// which is p(w) = cosθ / π
		return sameHemisphere(wo, wi) ? absCosTheta(wi) * Math::float_InvPi : 0;
	}

	auto BxDF::rho(Math::vec3 const& w, int nSamples, Math::point2 const* u) const noexcept -> Spectrum {
		Spectrum r(0.f);
		for (int i = 0; i < nSamples; ++i) {
			// Estimate one term of ρhd
			Math::vec3 wi;
			float pdf = 0;
			Spectrum f = sample_f(w, &wi, u[i], &pdf);
			if (pdf > 0) r += f * absCosTheta(wi) / pdf;
		}
		return r / nSamples;
	}
	
	auto BxDF::rho(int nSamples, Math::point2 const* u1, Math::point2 const* u2) const noexcept -> Spectrum {
		Spectrum r(0.f);
		for (int i = 0; i < nSamples; ++i) {
			// Estimate one term of ρhh
			Math::vec3 wo, wi;
			wo = uniformSampleHemisphere(u1[i]);
			float pdfo = uniformHemispherePdf(), pdfi = 0;
			Spectrum f = sample_f(wo, &wi, u2[i], &pdfi);
			if (pdfi > 0)
				r += f * absCosTheta(wi) * absCosTheta(wo) / (pdfo * pdfi);
		}
		return r / (Math::float_Pi * nSamples);
	}
}