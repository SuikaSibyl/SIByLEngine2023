module;
export module Tracer.BxDF:ScaledBxDF;
import :BxDF;
import SE.Math.Geometric;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	/**
	* ScaledBxDF takes a given BxDF and scale its contribution with a Spectrum value.
	* It is used by MixMaterial, which creates BSDFs based on a weighted combination 
	* of two other materials.
	*/
	export struct ScaledBxDF :public BxDF
	{
		ScaledBxDF(BxDF* bxdf, Spectrum const& scale)
			:BxDF(bxdf->type), bxdf(bxdf), scale(scale) {}

		virtual auto f(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> Spectrum override {
			return scale * bxdf->f(wo, wi);
		}

		virtual auto sample_f(Math::vec3 const& wo, Math::vec3* wi,
			Math::point2 const& sample, float* _pdf, Type* sampledType = nullptr) const noexcept -> Spectrum override {
			return scale * bxdf->sample_f(wo, wi, sample, _pdf, sampledType);
		}

		virtual auto rho(Math::vec3 const& wo, int nSamples, Math::point2 const* samples) const noexcept -> Spectrum override {
			return scale * bxdf->rho(wo, nSamples, samples);
		}

		virtual auto rho(int nSamples, Math::point2 const* u1, Math::point2 const* u2) const noexcept -> Spectrum override {
			return scale * bxdf->rho(nSamples, u1, u2);
		}

	private:
		BxDF* bxdf;
		Spectrum scale;
	};
}