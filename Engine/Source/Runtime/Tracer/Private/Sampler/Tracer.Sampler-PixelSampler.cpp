module;
#include <cstdint>
#include <vector>
module Tracer.Sampler:PixelSampler;
import Tracer.Sampler;
import SE.Math.Misc;
import SE.Math.Geometric;

namespace SIByL::Tracer
{
	PixelSampler::PixelSampler(int64_t samplesPerPixel, int nSampledDimensions)
		:Sampler(samplesPerPixel)
	{
		for (int i = 0; i < nSampledDimensions; ++i) {
			samples1D.push_back(std::vector<float>(samplesPerPixel));
			samples2D.push_back(std::vector<Math::point2>(samplesPerPixel));
		}
	}

	auto PixelSampler::startNextSample() noexcept -> bool {
		current1DDimension = current2DDimension = 0;
		return Sampler::startNextSample();
	}

	auto PixelSampler::setSampleNumber(int64_t sampleNum) noexcept -> bool {
		current1DDimension = current2DDimension = 0;
		return Sampler::setSampleNumber(sampleNum);
	}

	auto PixelSampler::get1D() noexcept -> float {
		if (current1DDimension < samples1D.size())
			return samples1D[current1DDimension++][currentPixelSampleIndex];
		else
			return rng.uniformFloat();
	}
	
	auto PixelSampler::get2D() noexcept -> Math::point2 {
		if (current2DDimension < samples2D.size())
			return samples2D[current2DDimension++][currentPixelSampleIndex];
		else
			return Math::point2{ rng.uniformFloat(), rng.uniformFloat() };
	}

}