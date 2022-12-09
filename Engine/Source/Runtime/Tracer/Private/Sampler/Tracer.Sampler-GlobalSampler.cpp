module;
#include <cstdint>
#include <vector>
#include <memory>
module Tracer.Sampler:GlobalSampler;
import Tracer.Sampler;
import SE.Core.Memory;
import SE.Math.Geometric;
import Tracer.Camera;

namespace SIByL::Tracer
{
	auto GlobalSampler::startPixel(Math::ipoint2 const& p) noexcept -> void {
		Sampler::startPixel(p);
		dimension = 0;
		intervalSampleIndex = getIndexForSample(0);
		// compute arrayEndDim for dimensions used for array samples
		arrayEndDim = arrayStartDim + sampleArray1D.size() + 2 * sampleArray2D.size();
		// compute 1D array samples for GlobalSampler
		for (size_t i = 0; i < samples1DArraySizes.size(); ++i) {
			int nSamples = samples1DArraySizes[i] * samplesPerPixel;
			for (int j = 0; j < nSamples; ++j) {
				int64_t index = getIndexForSample(j);
				sampleArray1D[i][j] =
					sampleDimension(index, arrayStartDim + i);
			}
		}
		// compute 2D array samples for GlobalSampler
		int dim = arrayStartDim + samples1DArraySizes.size();
		for (size_t i = 0; i < samples2DArraySizes.size(); ++i) {
			int nSamples = samples2DArraySizes[i] * samplesPerPixel;
			for (int j = 0; j < nSamples; ++j) {
				int64_t idx = getIndexForSample(j);
				sampleArray2D[i][j].x = sampleDimension(idx, dim);
				sampleArray2D[i][j].y = sampleDimension(idx, dim + 1);
			}
			dim += 2;
		}
	}

	auto GlobalSampler::startNextSample() noexcept -> bool {
		dimension = 0;
		intervalSampleIndex = getIndexForSample(currentPixelSampleIndex + 1);
		return Sampler::startNextSample();
	}

	auto GlobalSampler::setSampleNumber(int64_t sampleNum) noexcept -> bool {
		dimension = 0;
		intervalSampleIndex = getIndexForSample(sampleNum);
		return Sampler::setSampleNumber(sampleNum);
	}
	
	auto GlobalSampler::get1D() noexcept -> float {
		if (dimension >= arrayStartDim && dimension < arrayEndDim)
			dimension = arrayEndDim;
		return sampleDimension(intervalSampleIndex, dimension++);
	}
	
	auto GlobalSampler::get2D() noexcept -> Math::point2 {
		if (dimension + 1 >= arrayStartDim && dimension < arrayEndDim)
			dimension = arrayEndDim;
		Math::point2 p(sampleDimension(intervalSampleIndex, dimension),
			sampleDimension(intervalSampleIndex, dimension + 1));
		dimension += 2;
		return p;
	}

}