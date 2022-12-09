export module Tracer.BSSRDF:SeparableBSSRDF;
import SE.Math.Misc;
import SE.Math.Geometric;
import Tracer.Interactable;
import Tracer.Spectrum;
import Tracer.BxDF;

namespace SIByL::Tracer
{
	/**
	* BSSRDF
	*/
	export struct SeparableBSSRDF :public BSSRDF
	{
		SeparableBSSRDF(SurfaceInteraction const& po, float eta, Material const* material, TransportMode mode)
			: BSSRDF(po, eta), ns(po.shading.n), ss(Math::normalize(po.shading.dpdu))
			,  ts(Math::cross(ns, ss)), material(material), mode(mode) {}

		auto S(SurfaceInteraction const& pi, Math::vec3 const& wi) noexcept -> Spectrum;

		auto Sw(Math::vec3 const& w) const noexcept -> Spectrum {
			float c = 1 - 2 * fresnelMoment1(1.f / eta);
			return (1 - frDielectric(cosTheta(w), 1, eta)) / (c * Math::float_Pi);
		}

		auto Sp(SurfaceInteraction const& pi) const noexcept -> Spectrum {
			return Sr(Math::distance(po.p, pi.p));
		}

		virtual auto Sr(float d) const noexcept -> Spectrum = 0;

	private:
		Math::normal3 const ns;
		Math::vec3 const ss, ts;
		Material const* material;
		TransportMode const mode;
	};
};