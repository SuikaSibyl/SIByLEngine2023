module;
#include <cmath>
#include <filesystem>
export module Tracer.BxDF:FourierBSDF;
import :BxDF;
import Core.IO;
import Core.Memory;
import Math.Vector;
import Math.Geometry;
import Math.Limits;
import Math.Common;
import Tracer.Spectrum;

namespace SIByL::Tracer
{
	export inline auto Fourier(float const* a, int m, double cosPhi) noexcept -> float {
		double value = 0.0;
		// Initialize cosine iterates
		double cosKMinusOnePhi = cosPhi;
		double cosKPhi = 1;
		for (int k = 0; k < m; ++k) {
			// Add the current summandand update the cosine iterates
			value += a[k] * cosKPhi;
			double cosKPlusOnePhi = 2 * cosPhi * cosKPhi - cosKMinusOnePhi;
			cosKMinusOnePhi = cosKPhi;
			cosKPhi = cosKPlusOnePhi;
		}
		return value;
	}

	export inline auto CatmullRomWeights(int size, const float* nodes, float x, int* offset, float* weights) noexcept -> bool {
		// Return false if x is out of bounds 
		if (!(x >= nodes[0] && x <= nodes[size - 1]))
			return false;
		// Search for the interval idx containing x 
		int idx = Math::findInterval(size, [&](int i) { return nodes[i] <= x; });
		*offset = idx - 1;
		float x0 = nodes[idx], x1 = nodes[idx + 1];
		// Compute the t parameter and powers 
		float t = (x - x0) / (x1 - x0), t2 = t * t, t3 = t2 * t;
		// Compute initial node weights w1 and w2 
		weights[1] = 2 * t3 - 3 * t2 + 1;
		weights[2] = -2 * t3 + 3 * t2;
		// Compute first node weight w0 
		if (idx > 0) {
			float w0 = (t3 - 2 * t2 + t) * (x1 - x0) / (x1 - nodes[idx - 1]);
			weights[0] = -w0;
			weights[2] += w0;
		}
		else {
			float w0 = t3 - 2 * t2 + t;
			weights[0] = 0;
			weights[1] -= w0;
			weights[2] += w0;
		}
		// Compute last node weight 
		// TODO
		return true;
	}

	export struct FourierBSDFTable {
		static inline auto read(Core::filepath const& filename, FourierBSDFTable* table) noexcept -> bool;

		auto getAk(int offsetI, int offsetO, int* mptr) const noexcept-> float const* {
			*mptr = m[offsetO * nMu + offsetI];
			return a + aOffset[offsetO * nMu + offsetI];
		}
		auto getWeightsAndOffset(float cosTheta, int* offset, float weights[4]) const noexcept -> bool {
			return CatmullRomWeights(nMu, mu, cosTheta, offset, weights);
		}

		int nChannels;
		int nMu;
		float* mu;
		int* m;
		int* aOffset;
		float* a;
		float eta;
		int	mMax;
	};

	export struct FourierBSDF :public BxDF {
		FourierBSDF(FourierBSDFTable const& bsdfTable, TransportMode mode)
			: BxDF(BxDF::Type(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_GLOSSY))
			, bsdfTable(bsdfTable), mode(mode) {}

		virtual auto f(Math::vec3 const& wo, Math::vec3 const& wi) const noexcept -> Spectrum override {
			// Find the zenith angle cosinesand azimuth difference angle
			float muI = cosTheta(-wi), muO = cosTheta(wo);
			float cosPhi = cosDPhi(-wi, wo);
			// Compute Fourier coefficients ak for (μi, μo)
			//  Determine offsets and weights for μi and μo
			int offsetI, offsetO;
			float weightsI[4], weightsO[4];
			if (!bsdfTable.getWeightsAndOffset(muI, &offsetI, weightsI) ||
				!bsdfTable.getWeightsAndOffset(muO, &offsetO, weightsO))
				return Spectrum(0.f);
			//  Allocate storage to accumulate ak coefficients
			float* ak = Core::Alloca<float>(bsdfTable.mMax * bsdfTable.nChannels);
			memset(ak, 0, bsdfTable.mMax * bsdfTable.nChannels * sizeof(float));
			//  Accumulate weighted sums of nearby ak coefficients
			int mMax = 0;
			for (int b = 0; b < 4; ++b) {
				for (int a = 0; a < 4; ++a) {
					// Add contribution of(a, b) to ak values
					float weight = weightsI[a] * weightsO[b];
					if (weight != 0) {
						int m;
						float const* ap = bsdfTable.getAk(offsetI + a, offsetO + b, &m);
						mMax = std::max(mMax, m);
						for (int c = 0; c < bsdfTable.nChannels; ++c)
							for (int k = 0; k < m; ++k)
								ak[c * bsdfTable.mMax + k] += weight * ap[c * m + k];
					}
				}
			}
			// Evaluate Fourier expansion for angle φ
			float Y = std::max(0.f, Fourier(ak, mMax, cosPhi));
			float scale = muI != 0 ? (1 / std::abs(muI)) : 0.f;
			// Update scale to account for adjoint light transport
			if (bsdfTable.nChannels == 1)
				return Spectrum(Y * scale);
			else {
				// Computeand return RGB colors for tabulated BSDF
				float R = Fourier(ak + 1 * bsdfTable.mMax, mMax, cosPhi);
				float B = Fourier(ak + 2 * bsdfTable.mMax, mMax, cosPhi);
				float G = 1.39829f * Y - 0.100913f * B - 0.297375f * R;
				float rgb[3] = { R * scale, G * scale, B * scale };
				return Spectrum::fromRGB(rgb).clamp();
			}
		}

	private:
		FourierBSDFTable const& bsdfTable;
		TransportMode const mode;
	};
}