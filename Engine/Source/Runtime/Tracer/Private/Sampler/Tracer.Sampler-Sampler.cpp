module;
#include <cstdint>
#include <vector>
#include <memory>
module Tracer.Sampler:Sampler;
import Tracer.Sampler;
import Core.Memory;
import Math.Geometry;
import Math.Random;
import Math.Limits;
import Tracer.Camera;

namespace SIByL::Tracer
{
	auto Sampler::getCameraSample(Math::ipoint2 const& pRaster) noexcept -> CameraSample {
		CameraSample cs;
		cs.pFilm = (Math::point2)pRaster + get2D();
		cs.time = get1D();
		cs.pLens = get2D();
		return cs;
	}
	
	auto Sampler::startPixel(Math::ipoint2 const& p) noexcept -> void {
		currentPixel = p;
		currentPixelSampleIndex = 0;
		// reset array offsets for next pixel sample
		array1DOffset = array2DOffset = 0;
	}
	
	auto Sampler::startNextSample() noexcept -> bool {
		// reset array offsets for next pixel sample
		array1DOffset = array2DOffset = 0;
		return ++currentPixelSampleIndex < samplesPerPixel;
	}
	
	auto Sampler::setSampleNumber(int64_t sampleNum) noexcept -> bool {
		// reset array offsets of next pixel sample
		array1DOffset = array2DOffset = 0;
		currentPixelSampleIndex = sampleNum;
		return currentPixelSampleIndex < samplesPerPixel;
	}

	auto Sampler::request1DArray(int n) noexcept -> void {
		samples1DArraySizes.emplace_back(n);
		sampleArray1D.emplace_back(std::vector<float>(n * samplesPerPixel));
	}

	auto Sampler::request2DArray(int n) noexcept -> void {
		samples2DArraySizes.emplace_back(n);
		sampleArray2D.emplace_back(std::vector<Math::point2>(n * samplesPerPixel));
	}

	auto Sampler::get1DArray(int n) noexcept -> float const* {
		if (array1DOffset == sampleArray1D.size())
			return nullptr;
		return &sampleArray1D[array1DOffset++][currentPixelSampleIndex * n];
	}

	auto Sampler::get2DArray(int n) noexcept -> Math::point2 const* {
		if (array2DOffset == sampleArray2D.size())
			return nullptr;
		return &sampleArray2D[array2DOffset++][currentPixelSampleIndex * n];
	}

	auto Sampler::roundCount(int n) const noexcept -> int {
		return n;
	}

	inline auto latinHypercube(float* samples, int nSamples, int nDim, Math::RNG& rng) noexcept -> void {
		// generate LHS samples along diagonal
		float invNSamples = 1.f / nSamples;
		for (int i = 0; i < nSamples; ++i)
			for (int j = 0; j < nDim; ++j) {
				float sj = (i + (rng.uniformFloat())) * invNSamples;
				samples[nDim * i + j] = std::min(sj, Math::one_minus_epsilon);
			}
		// permute LHS samples in each dimensions
		for (int i = 0; i < nDim; ++i) {
			for (int j = 0; j < nSamples; ++j) {
				int other = j + rng.uniformUInt32(nSamples - j);
				std::swap(samples[nDim * j + i], samples[nDim * other + i]);
			}
		}
	}

}