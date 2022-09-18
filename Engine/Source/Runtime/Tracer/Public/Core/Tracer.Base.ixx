module;
#include <vector>
#include <memory>
#include <cstdint>
#include <algorithm>
export module Tracer.Base;
import Core.Memory;
import Math.Geometry;
import Math.Random;

namespace SIByL::Tracer
{
	export struct CameraSample {
		Math::point2 pFilm;
		Math::point2 pLens;
		float time;
	};

	/**
	* Sample is used to generate a sequence of n-dimensional samples in [0,1)^n,
	* where one such sample vector is generated for each image sample and
	* the number of dimensions n in each sample may vary, depending on the calculations
	* performed by the light transport algorithms.
	*
	* Sample Vectors assumptions:
	* - First five dimensions generated are generallly used by Camera. The first two
	*   are specifically used to choose a point on the image inside the current pixel
	*   area; the third is used to compute the time; the fourth and fifth give a (u,v)
	*   lens position for depth of field.
	* - Some sampling algorithms generate better samples in some dimensions than in others.
	*   We assume in general, the earlier dimensions have the most well-placed sample value.
	*/
	export struct Sampler
	{
		/** Initialize with the number of samples that will be generated for each pixel */
		Sampler(int64_t spp) :samplesPerPixel(spp) {}

		/**
		* Providing the coordinates of the pixel in the image. Some implementations
		* use the knowledge of the pixel coordinate to improve overall distribution.
		*/
		virtual auto startPixel(Math::ipoint2 const& p) noexcept -> void;
		/**
		* Notify the sampler that subsequent request should return
		* the first dimension of the next sample
		*/
		virtual auto startNextSample() noexcept -> bool;
		/**
		* Allows integrators to set the index of the samples in
		* the current pixels to generate next
		*/
		virtual auto setSampleNumber(int64_t sampleNum) noexcept -> bool;

		/** Get a new instance of an initial Sampler */
		virtual auto clone(int seed) noexcept -> Scope<Sampler> = 0;

		/*
		* User of sample values should always requests sample dimensions in the same order.
		* Or the distribution would probably get some problem.
		*/

		/** sample value for the next dimension of the current sample vector */
		virtual auto get1D() noexcept -> float = 0;
		/** sample value for the next two dimensions of the current sample vector */
		virtual auto get2D() noexcept -> Math::point2 = 0;
		/** initialize camera sample for a given pixel */
		auto getCameraSample(Math::ipoint2 const& pRaster) noexcept -> CameraSample;

		/**
		* Provide arrays of samples, which might be better distributed.
		* Should be called before rendering begins
		*/
		auto request1DArray(int n) noexcept -> void;
		auto request2DArray(int n) noexcept -> void;
		/** get a pointer to the start of previously requested array of samples */
		auto get1DArray(int n) noexcept -> float const*;
		auto get2DArray(int n) noexcept -> Math::point2 const*;

		/**
		* Adjust the number of samples to a better number,
		* which could result in better distribution.
		*/
		virtual auto roundCount(int n) const noexcept -> int;

		/** sample count in a single pixel */
		int64_t const samplesPerPixel;
		/** which pixel is being sampled */
		Math::ipoint2 currentPixel;
		/** how many dimensions of the sample have been used */
		int64_t currentPixelSampleIndex;

	protected:
		/** The sizes of the requested sample arrays */
		std::vector<int> samples1DArraySizes, samples2DArraySizes;
		/** The memory for an entire pixel's worth of array samples */
		std::vector<std::vector<float>> sampleArray1D;
		/** The memory for an entire pixel's worth of array samples */
		std::vector<std::vector<Math::point2>> sampleArray2D;
		/** hold the index of the next array to return for the sample vector */
		size_t array1DOffset, array2DOffset;
	};

	/** Randomly permutes an array of count sample values, each of which has nDimensions dimensions. */
	export template<class T>
		inline auto shuffle(T* samp, int count, int nDimensions, Math::RNG& rng) noexcept -> void {
		for (int i = 0; i < count; ++i) {
			int other = i + rng.uniformUInt32(count - i);
			for (int j = 0; j < nDimensions; ++j)
				std::swap(samp[nDimensions * i + j], samp[nDimensions * other + j]);
		}
	}

	export inline auto latinHypercube(float* samples, int nSamples, int nDim, Math::RNG& rng) noexcept -> void;
}