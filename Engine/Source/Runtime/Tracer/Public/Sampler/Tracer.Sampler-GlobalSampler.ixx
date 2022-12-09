module;
#include <cstdint>
#include <vector>
#include <memory>
export module Tracer.Sampler:GlobalSampler;
import SE.Core.Memory;
import SE.Math.Geometric;
import Tracer.Base;
import Tracer.Camera;

namespace SIByL::Tracer
{
	/**
	* GlobalSampler generate consecutive samples that are spread across the entire image,
	* visiting completely different pixels in succession.
	*/
	export struct GlobalSampler :public Sampler
	{
		GlobalSampler(int64_t samplerPerPixel) :Sampler(samplerPerPixel) {}

		/**
		* Perform the inverse mapping from the current pixel and given sample index to a
		* global index into the overall set of sample vectors.
		*/
		virtual auto getIndexForSample(int64_t sampleNum) const noexcept -> int64_t = 0;

		/*
		* Returns the sample value for the given dimension of the indexth sample vector
		* in the sequence.
		* Specially the first two values are sample offset within the current pixel.
		*/
		virtual auto sampleDimension(int64_t index, int dimension) const noexcept -> float = 0;

		virtual auto startPixel(Math::ipoint2 const& p) noexcept -> void override;

		virtual auto startNextSample() noexcept -> bool override;

		virtual auto setSampleNumber(int64_t sampleNum) noexcept -> bool override;

		/** sample value for the next dimension of the current sample vector */
		virtual auto get1D() noexcept -> float override;
		/** sample value for the next two dimensions of the current sample vector */
		virtual auto get2D() noexcept -> Math::point2 override;

	private:
		/**
		* tracks the next dimension that the sampler implementation will be asked to 
		* generate a sample value for. It is incremented as get1D() & get2D() are called.
		*/
		int dimension;

		/** records the index of sample that corresponds to the current sample s_i in the current pixel */
		int64_t intervalSampleIndex;

		/** the first dimensions up to arrayStartDim are devoted to regular 1D and 2D samples */
		static int const arrayStartDim = 5;

		/** higher dimensions starting at arrayEndDim are used for further non-array 1D and 2D samples */
		int arrayEndDim;
	};
}