module;
#include <cmath>
#include <functional>
export module Tracer.BxDF:OrenNayar;
import :BxDF;
import SE.Math.Misc;
import SE.Math.Geometric;
import Tracer.Spectrum;

namespace SIByL::Tracer

{	/**
	* A microfacet diffuse reflection module by Oren & Nayar (1994).
	* Describe rough surface by V-shaped microfacets described by a spherical Gaussian distribution
	* with a single parameter σ, the standard deviation of the microfacet orientation angle.
	*/
	export struct OrenNayar :public BxDF
	{
		OrenNayar(Spectrum const& R, float _sigma)
			: BxDF(BxDF::Type(BSDF_REFLECTION | BSDF_DIFFUSE)), R(R)
			, A([&]() { float const sigma = Math::radians(_sigma);
						float const sigma2 = sigma * sigma;
						return 1.f - (sigma2 / (2.f * (sigma2 + 0.33f))); }())
			, B([&]() { float const sigma = Math::radians(_sigma);
						float const sigma2 = sigma * sigma;
						return 0.45f * sigma2 / (sigma2 + 0.09f); }())
		{}

		/** return the value of the distribution function for the given pair of directions. */
		virtual auto f(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> Spectrum override {
			float const sinThetaI = sinTheta(wi);
			float const sinThetaO = sinTheta(wo);
			// compute cosine term of Oren–Nayar model
			float const maxCos = [&]() {
				//  cos(a − b) = cosa cosb + sina sinb
				if (sinThetaI > 1e-4 && sinThetaO > 1e-4) {
					float const sinPhiI = sinPhi(wi), cosPhiI = cosPhi(wi);
					float const sinPhiO = sinPhi(wo), cosPhiO = cosPhi(wo);
					float const dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
					return std::max(0.f, dCos); }
				else return 0.f;
			}();
			// compute sine and tangent terms of Oren–Nayar model
			auto const [sinAlpha, tanBeta] = [&]() {
				if (absCosTheta(wi) > absCosTheta(wo))
					return std::pair<float, float>(sinThetaO, sinThetaI / absCosTheta(wi));
				else
					return std::pair<float, float>(sinThetaI, sinThetaO / absCosTheta(wo));
			}();
			return R * Math::float_InvPi * (A + B * maxCos * sinAlpha * tanBeta);
		}

	private:
		Spectrum const R;
		float const A, B;
	};
}