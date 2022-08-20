module;
#include <cmath>
#include <vector>
module Tracer.Sampler:StratifiedSampler;
import Tracer.Sampler;
import Math.Limits;
import Math.Vector;
import Math.Geometry;
import Math.Random;

namespace SIByL::Tracer
{
	StratifiedSampler::StratifiedSampler(int xPixelSamples, int yPixelSamples, bool jitterSamples, int nSampledDimensions)
		: PixelSampler(xPixelSamples* yPixelSamples, nSampledDimensions)
		, xPixelSamples(xPixelSamples), yPixelSamples(yPixelSamples)
		, jitterSamples(jitterSamples)
	{}

	auto StratifiedSampler::startPixel(Math::ipoint2 const& p) noexcept -> void {
		// generate single stratified samples for the pixel
		for (size_t i = 0; i < samples1D.size(); ++i) {
			stratifiedSample1D(&samples1D[i][0], xPixelSamples * yPixelSamples,
				rng, jitterSamples);
			shuffle(&samples1D[i][0], xPixelSamples * yPixelSamples, 1, rng);
		}
		for (size_t i = 0; i < samples2D.size(); ++i) {
			stratifiedSample2D(&samples2D[i][0], xPixelSamples, yPixelSamples,
				rng, jitterSamples);
			shuffle(&samples2D[i][0], xPixelSamples * yPixelSamples, 1, rng);
		}
		// generate arrays of stratified samples for the pixel
		for (size_t i = 0; i < samples1DArraySizes.size(); ++i)
			for (int64_t j = 0; j < samplesPerPixel; ++j) {
				int count = samples1DArraySizes[i];
				stratifiedSample1D(&sampleArray1D[i][j * count], count, rng,
					jitterSamples);
				shuffle(&sampleArray1D[i][j * count], count, 1, rng);
			}
		for (size_t i = 0; i < samples2DArraySizes.size(); ++i)
			for (int64_t j = 0; j < samplesPerPixel; ++j) {
				int count = samples2DArraySizes[i];
				latinHypercube(&sampleArray2D[i][j * count].x, count, 2, rng);
			}
		PixelSampler::startPixel(p);
	}

	inline auto stratifiedSample1D(float* samp, int nSamples, Math::RNG& rng, bool jitter) noexcept -> void {
		float const invNSamples = 1.f / nSamples;
		for (int i = 0; i < nSamples; ++i) {
			float delta = jitter ? rng.uniformFloat() : 0.5f;
			samp[i] = std::min((i + delta) * invNSamples, Math::one_minus_epsilon);
		}
	}

	inline auto stratifiedSample2D(Math::point2* samp, int nx, int ny, Math::RNG& rng, bool jitter) noexcept -> void {
		float const dx = 1.f / nx, dy = 1.f / ny;
		for (int y = 0; y < ny; ++y)
			for (int x = 0; x < nx; ++x) {
				float jx = jitter ? rng.uniformFloat() : 0.5f;
				float jy = jitter ? rng.uniformFloat() : 0.5f;
				samp->x = std::min((x + jx) * dx, Math::one_minus_epsilon);
				samp->y = std::min((y + jy) * dy, Math::one_minus_epsilon);
				++samp;
			}
	}

}