module;
#include <cmath>
#include <memory>
#include <vector>
export module Tracer.Texture:MIPMap;
import Core.Memory;
import Math.Common;
import Math.Vector;
import Math.Geometry;
import Parallelism.Parallel;

namespace SIByL::Tracer
{
	export enum struct ImageWrap {
		REPEAT,
		BLACK,
		CLAMP,
	};

	struct ResampleWeight {
		int firstTexel;
		float weight[4];
	};

	auto Lanczos(float, float tau = 2) noexcept -> float {
		return 1.f;
	}

	export template <class T> struct MIPMap
	{
		MIPMap(Math::ipoint2 const& res, T const* img, bool doTrilinear,
			float maxAnistropy, ImageWrap wrapMode)
			: doTrilinear(doTrilinear), maxAnisotropy(maxAnisotropy)
			, wrapMode(wrapMode), resolution(res)
		{
			std::unique_ptr<T[]> resampledImage = nullptr;
			if (!Math::isPowerOf2(resolution[0]) || !Math::isPowerOf2(resolution[1])) {
				// resample image to power-of-two resoltion
				Math::ipoint2 resPow2(Math::roundUpPow2(resolution[0]), Math::roundUpPow2(resolution[1]));
				//  Resample image in s direction
				std::unique_ptr<ResampleWeight[]> sWeights = resampleWeights(resolution[0], resPow2[0]);
				resampledImage.reset(new T[resPow2[0] * resPow2[1]]);
				//  Apply sWeights to zoom in s direction
				Parallelism::ParallelFor([&](int t) {
						for (int s = 0; s < resPow2[0]; ++s) {
							//  Compute texel(s, t) in s - zoomed image
							resampledImage[t * resPow2[0] + s] = 0.f;
							for (int j = 0; j < 4; ++j) {
								int origS = sWeights[s].firstTexel + j;
								if (wrapMode == ImageWrap::REPEAT)
									origS = Math::mod(origS, resolution[0]);
								else if (wrapMode == ImageWrap::CLAMP)
									origS = Math::clamp(origS, 0, resolution[0] - 1);
								if (origS >= 0 && origS < (int)resolution[0])
									resampledImage[t * resPow2[0] + s] +=
									sWeights[s].weight[j] * img[t * resolution[0] + origS];
							}
						}
					}, resolution[1], 16);
				//  Resample image in t direction
				std::unique_ptr<ResampleWeight[]> tWeights = resampleWeights(resolution[1], resPow2[1]);


				resolution = resPow2;
			}
			// intiialize levels of MIPMap from images
			int nLevels = 1 + Math::log2Int(std::max(resolution[0], resolution[1]));
			pyramid.resize(nLevels);
			//  Initialize most detailed level of MIPMap
				for (int i = 1; i < nLevels; ++i) {
					//  Initialize ith MIPMap level from i − 1st level
				}
			// intialize EWA filter weights if needed
		}

		auto width() const noexcept -> int { return resolution.x; }
		auto height() const noexcept -> int { return resolution.y; }
		auto levels() const noexcept -> int { return pyramid.size(); }

		auto texel(int level, int s, int t) const noexcept -> T const& {
			Core::BlockedArray<T> const& l = *pyramid[level];
			// compute texel (s,t) accounting for boundary conditions
			switch (wrapMode) {
			case ImageWrap::REPEAT:
				s = Mod(s, l.uSize());
				t = Mod(t, l.vSize());
				break;
			case ImageWrap::CLAMP:
				s = Clamp(s, 0, l.uSize() - 1);
				t = Clamp(t, 0, l.vSize() - 1);
				break;
			case ImageWrap::BLACK: {
				static const T black = 0.f;
				if (s < 0 || s >= (int)l.uSize() ||
					t < 0 || t >= (int)l.vSize())
					return black;
				break; }
			}
			return l(s, t);
		}

	private:
		auto resampleWeights(int oldRes, int newRes) noexcept -> std::unique_ptr<ResampleWeight[]> {
			std::unique_ptr<ResampleWeight[]> wt(new ResampleWeight[newRes]);
			float filterwidth = 2.f;
			for (int i = 0; i < newRes; ++i) {
				//  Compute image resampling weights for ith texel
				float center = (i + .5f) * oldRes / newRes;
				wt[i].firstTexel = std::floor((center - filterwidth) + 0.5f);
				for (int j = 0; j < 4; ++j) {
					float pos = wt[i].firstTexel + j + .5f;
					wt[i].weight[j] = Lanczos((pos - center) / filterwidth);
				}
				//  Normalize filter weights for texel resampling
				float invSumWts = 1 / (wt[i].weight[0] + wt[i].weight[1] + wt[i].weight[2] + wt[i].weight[3]);
				for (int j = 0; j < 4; ++j)
					wt[i].weight[j] *= invSumWts;
			}
			return wt;
		}

		bool	  const doTrilinear;
		float	  const maxAnisotropy;
		ImageWrap const wrapMode;
		Math::ipoint2	resolution;
		std::vector<std::unique_ptr<Core::BlockedArray<T>>> pyramid;
	};
}