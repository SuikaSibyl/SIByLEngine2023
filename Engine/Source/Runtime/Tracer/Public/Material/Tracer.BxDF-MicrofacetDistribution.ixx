module;
#include <cmath>
export module Tracer.BxDF:MicrofacetDistribution;
import :BxDF;
import SE.Math.Misc;
import SE.Math.Geometric;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	/**
	* Based on the idea that rough surface can be modeled as a collection of small microfacets.
	* Statically model the distribution of facet orientations.
	* MicrofacetDistribution is an interface that encapsulate facets' geometric properties
	*/
	export struct MicrofacetDistribution {
		MicrofacetDistribution(bool sampleVisibleArea)
			:sampleVisibleArea(sampleVisibleArea) {}

		/**
		* Describe the normal distribution property of characterize the microsurface for rendering.
		* Distribution function D(wh) gives the differential area of microfacets with the surface normal wh.
		* The function is defined in the same BSDF coordinate system as BxDFs.
		* It is always normalized to be physically plausible, which means ∫(D*cosθ) dω = 1
		* @return the differential area of microfacets oriented with the given normal vector ω.
		*/
		virtual auto D(Math::vec3 const& wh) const noexcept -> float = 0;
		
		/**
		* Shadowing-masking functions are traditionally expressed in terms of an auxiliary function Λ(ω).
		* Measures invisible masked microfacet area per visible microfacet area.
		* Supporting the uniform implementation of G1.
		*/
		virtual auto lambda(Math::vec3 const& w) const noexcept -> float = 0;

		/**
		* Describe the masking-shadowing property of characterize the microsurface for rendering.
		* Using Smith's masking-shadowing function G1(ω,ωh)
		* @return the fraction of microfacets with normal ωh that are visible from direction ω, ∈[0,1]
		*/
		auto G1(Math::vec3 const& w) const noexcept -> float { 
			return 1.f / (1.f + lambda(w)); 
		}

		/**
		* G(ωo,ωi) gives further geometric properties of a microfacet distribution.
		* Assuming microfacet visibility is more likely the higher up a given point on a microfacet is.
		* @return the fraction of microfacets in a differential area that are visible from both directions ωo & ωi
		*/
		auto G(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> float {
			return 1.f / (1.f + lambda(wo) + lambda(wi));
		}

		/**
		* 
		*/
		virtual auto sample_wh(Math::vec3 const& wo, Math::point2 const& u) const noexcept -> Math::vec3 = 0;

		auto pdf(Math::vec3 const& wo, Math::vec3 const& wh) const noexcept -> float {
			if (sampleVisibleArea)
				return D(wh) * G1(wo) * absDot(wo, wh) / absCosTheta(wo);
			else
				return D(wh) * absCosTheta(wh);
		}

	protected:
		bool const sampleVisibleArea;
	};

	/**
	* A widely used microfacet distribution function based on 
	* a Gaussian distribution of microfacet slopes.
	* Which is due to Beckmann & Spizzichino (1963).
	*/
	export struct BeckmannDistribution :public MicrofacetDistribution {
		BeckmannDistribution(float alphax, float alphay, bool sampleVisibleArea)
			: MicrofacetDistribution(sampleVisibleArea), alphax(alphax), alphay(alphay) {}

		/**
		* Distribution function D(wh) gives the differential area of microfacets with the surface normal wh.
		* The function is defined in the same BSDF coordinate system as BxDFs.
		* It is always normalized to be physically plausible, which means ∫(D*cosθ) dω = 1
		* @return the differential area of microfacets oriented with the given normal vector ω.
		*/
		virtual auto D(Math::vec3 const& wh) const noexcept -> float override {
			float const tan2theta = tan2Theta(wh);
			// handle infinite value of tan2theta specially
			// which is actually valid at perfectly grazing directions.
			if (std::isinf(tan2theta)) return 0.f;
			float const cos4Theta = cos2Theta(wh) * cos2Theta(wh);
			return std::exp(-tan2theta * (cos2Phi(wh) / (alphax * alphax) + sin2Phi(wh) / (alphay * alphay)))
				/ (Math::float_Pi * alphax * alphay * cos4Theta);
		}

		/**
		* Specify the BRDF's roughness with a scalar parameter in [0,1] rather than
		* directly specifying alpha value, it could be more convenient.
		* Where values close to zero correspond to near-perfect specular reflection.
		* roughnessToAlpha performs a mapping from such roughness to alpha values.
		*/
		static inline auto roughnessToAlpha(float roughness) noexcept -> float {
			roughness = std::max(roughness, (float)1e-3);
			float const x = std::log(roughness);
			float const x2 = x * x;
			return 1.62142f + 0.819955f * x + 0.1734f * x2 + 0.0171201f * x2 * x + 0.000640711f * x2 * x2;
		}

		/**
		* Shadowing-masking functions are traditionally expressed in terms of an auxiliary function Λ(ω).
		* Measures invisible masked microfacet area per visible microfacet area.
		* Supporting the uniform implementation of G1.
		*/
		virtual auto lambda(Math::vec3 const& w) const noexcept -> float override {
			float const absTanTheta = std::abs(tanTheta(w));
			if (std::isinf(absTanTheta)) return 0.f;
			// Compute alpha for direction w
			float const alpha = std::sqrt(cos2Phi(w) * alphax * alphax + sin2Phi(w) * alphay * alphay);
			float const a = 1.f / (alpha * absTanTheta);
			if (a >= 1.6f) return 0;
			return (1 - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
		}

		virtual auto sample_wh(Math::vec3 const& wo, Math::point2 const& u) const noexcept -> Math::vec3 override {
			if (!sampleVisibleArea) {
				// Sample full distribution of normals for Beckmann distribution
				//  Compute tan2 θand φ for Beckmann distribution sample
				float tan2Theta, phi;
				if (alphax == alphay) {
					float logSample = std::log(u[0]);
					if (std::isinf(logSample)) logSample = 0;
					tan2Theta = -alphax * alphax * logSample;
					phi = u[1] * 2 * Math::float_Pi;
				}
				else {
					// Compute tan2Thetaand phi for anisotropic Beckmann distribution
				}
				//  Map sampled Beckmann angles to normal direction wh
				float cosTheta = 1 / std::sqrt(1 + tan2Theta);
				float sinTheta = std::sqrt(std::max((float)0, 1 - cosTheta * cosTheta));
				Math::vec3 wh = Math::sphericalDirection(sinTheta, cosTheta, phi);
				if (!sameHemisphere(wo, wh)) wh = -wh;
				return wh;
			}
			else {
				// Sample visible area of normals for Beckmann distribution
			}
		}

	private:
		/**
		* Used for an anisotropic distribution, where normal distribution
		* also varies depending on the azimuthal oriented of w_h.
		* microfacets oriented perpendicular to the x axis 
		*/
		float const alphax;
		/**
		* Used for an anisotropic distribution, where normal distribution
		* also varies depending on the azimuthal oriented of w_h.
		* microfacets oriented perpendicular to the y axis 
		*/
		float const alphay;
	};

	/**
	* A widely used microfacet distribution module.
	* It falls off to zero more slowly for directions far from the surface well.
	* Which matches the properties of many real-world surfaces well.
	*/
	export struct TrowbridgeReitzDistribution :public MicrofacetDistribution {
		/**
		* Distribution function D(wh) gives the differential area of microfacets with the surface normal wh.
		* The function is defined in the same BSDF coordinate system as BxDFs.
		* It is always normalized to be physically plausible, which means ∫(D*cosθ) dω = 1
		* @return the differential area of microfacets oriented with the given normal vector ω.
		*/
		virtual auto D(Math::vec3 const& wh) const noexcept -> float override {
			float tan2theta = tan2Theta(wh);
			if (std::isinf(tan2theta)) return 0.;
			float const cos4Theta = cos2Theta(wh) * cos2Theta(wh);
			float e = (cos2Phi(wh) / (alphax * alphax) + sin2Phi(wh) / (alphay * alphay)) * tan2theta;
			return 1.f / (Math::float_Pi * alphax * alphay * cos4Theta * (1 + e) * (1 + e));
		}

		/**
		* Specify the BRDF's roughness with a scalar parameter in [0,1] rather than
		* directly specifying alpha value, it could be more convenient.
		* Where values close to zero correspond to near-perfect specular reflection.
		* roughnessToAlpha performs a mapping from such roughness to alpha values.
		*/
		static inline auto roughnessToAlpha(float roughness) noexcept -> float {
			roughness = std::max(roughness, (float)1e-3);
			float const x = std::log(roughness);
			float const x2 = x * x;
			return 1.62142f + 0.819955f * x + 0.1734f * x2 + 0.0171201f * x2 * x + 0.000640711f * x2 * x2;
		}

		/**
		* Shadowing-masking functions are traditionally expressed in terms of an auxiliary function Λ(ω).
		* Measures invisible masked microfacet area per visible microfacet area.
		* Supporting the uniform implementation of G1.
		*/
		virtual auto lambda(Math::vec3 const& w) const noexcept -> float override {
			float absTanTheta = std::abs(tanTheta(w));
			if (std::isinf(absTanTheta)) return 0.f;
			// Compute alpha for direction w
			float alpha = std::sqrt(cos2Phi(w) * alphax * alphax + sin2Phi(w) * alphay * alphay);
			float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
			return (-1 + std::sqrt(1.f + alpha2Tan2Theta)) / 2;
		}

	private:
		/**
		* Used for an anisotropic distribution, where normal distribution
		* also varies depending on the azimuthal oriented of w_h.
		* microfacets oriented perpendicular to the x/y axis
		*/
		float const alphax, alphay;
	};
}