module;
#include <cstdint>
#include <vector>
export module Tracer.Sampler:PixelSampler;
import Tracer.Ray;
import Math.Geometry;
import Math.Random;

namespace SIByL::Tracer
{
	/**
	* PixelSampler implements some functionality that is useful for the implementation
	* of some sampling algorithms, which can naturally generate all of the dimensions'
	* sample values for all of the sample vectors for a pixel at the same time.
	*/
	export struct PixelSampler :public Sampler
	{
		PixelSampler(int64_t samplesPerPixel, int nSampledDimensions);

		virtual auto startNextSample() noexcept -> bool override;
		virtual auto setSampleNumber(int64_t sampleNum) noexcept -> bool override;
		
		virtual auto get1D() noexcept -> float override;
		virtual auto get2D() noexcept -> Math::point2 override;

	protected:
		/**
		* For each precomputed dimension, the constructor allocates a vector
		* to store samples values, with one value for each sample in the pixel.
		* They should be filled in startPixel() methods by derived classes.
		*/
		std::vector<std::vector<float>>			samples1D;
		std::vector<std::vector<Math::point2>>	samples2D;
		/** stores the offeset into the respective arrays for the current pixel samples */
		int current1DDimension = 0, current2DDimension = 0;

		Math::RNG rng;
	};
}