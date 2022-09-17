module;
#include <memory>
export module Tracer.BSSRDF:TabulatedBSSRDF;
import :SeparableBSSRDF;
import Math.Vector;
import Math.Geometry;
import Math.Trigonometric;
import Tracer.Interactable;
import Tracer.Spectrum;
import Tracer.BxDF;

namespace SIByL::Tracer
{
	export struct BSSRDFTable {
		inline auto evalProfile(int rhoIndex, int radiusIndex) const noexcept -> float {
			return profile[rhoIndex * nRadiusSamples + radiusIndex];
		}

		int const nRhoSamples, nRadiusSamples;
		std::unique_ptr<float[]> rhoSamples, radiusSamples;
		std::unique_ptr<float[]> profile;
		std::unique_ptr<float[]> rhoEff;
	};

	export struct TabulatedBSSRDF :public SeparableBSSRDF
	{
		TabulatedBSSRDF(SurfaceInteraction const& po, Material const* material, TransportMode mode, float eta,
			Spectrum const& sigma_a, Spectrum const& sigma_s, BSSRDFTable const& table)
			: SeparableBSSRDF(po, eta, material, mode), table(table) 
		{
			sigma_t = sigma_a + sigma_s;
			for (int c = 0; c < Spectrum::nSamples; ++c)
				rho[c] = sigma_t[c] != 0 ? (sigma_s[c] / sigma_t[c]) : 0;
		}

		virtual auto Sr(float r) const noexcept -> Spectrum override {
			Spectrum Sr(0.f);
			for (int ch = 0; ch < Spectrum::nSamples; ++ch) {
				// Convert r into unitless optical radius roptical
				float rOptical = r * sigma_t[ch];
				// Compute spline weights to interpolate BSSRDF on channel ch
				int rhoOffset, radiusOffset;
				float rhoWeights[4], radiusWeights[4];
				if (!CatmullRomWeights(table.nRhoSamples, table.rhoSamples.get(),
					rho[ch], &rhoOffset, rhoWeights) ||
					!CatmullRomWeights(table.nRadiusSamples, table.radiusSamples.get(),
						rOptical, &radiusOffset, radiusWeights))
					continue;
				// Set BSSRDF value Sr[ch] using tensor spline interpolation
				float sr = 0;
				for (int i = 0; i < 4; ++i) {
					for (int j = 0; j < 4; ++j) {
						float weight = rhoWeights[i] * radiusWeights[j];
						if (weight != 0)
							sr += weight * table.evalProfile(rhoOffset + i, radiusOffset + j);
					}
				}
				//  Cancel marginal PDF factor from tabulated BSSRDF profile 700
				if (rOptical != 0)
					sr /= 2 * Math::float_Pi * rOptical;
				Sr[ch] = sr;
			}
			// Transform BSSRDF value into world space units
			Sr *= sigma_t * sigma_t;
			return Sr.clamp();
		}

	private:
		BSSRDFTable const& table;
		Spectrum sigma_t, rho;
	};
}