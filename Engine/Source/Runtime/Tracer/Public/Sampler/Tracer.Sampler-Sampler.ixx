module;
#include <cstdint>
#include <memory>
export module Tracer.Sampler:Sampler;
import Math.Geometry;
import Tracer.Camera;

namespace SIByL::Tracer
{
	export struct Sampler
	{
		/** Initialize with the number of samples that will be generated for each pixel */
		Sampler(int64_t spp) :samplesPerPixel(spp) {}

		/** 
		* Providing the coordinates of the pixel in the image. Some implementations
		* use the knowledge of the pixel coordinate to improve overall distribution.
		*/
		virtual auto startPixel(Math::ipoint2 const& p) noexcept -> void = 0;
		/**
		* Notify the sampler that subsequent request should return 
		* the first dimension of the next sample
		*/
		virtual auto startNextSample() noexcept -> bool = 0;

		/** Get a new instance of an initial Sampler */
		virtual auto clone(int seed) noexcept -> std::unique_ptr<Sampler> = 0;

		/**
		* Allows integrators to set the index of the samples in 
		* the current pixels to generate next
		*/
		virtual auto setSampleNumber(int64_t sampleNum) noexcept -> bool = 0;

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
		* Provide arrays of samples, which might be better distributed
		* Should be called before rendering begins
		*/
		auto request1DArray(int n) noexcept -> void;
		auto request2DArray(int n) noexcept -> void;
		/** get a pointer to the start of previously requested array of samples */
		auto get1DArray(int n) const noexcept -> float*;
		auto get2DArray(int n) const noexcept -> Math::point2*;

		/**
		* adjust the number of samples to a better number, 
		* which could result in better distribution��
		*/
		virtual auto roundCount(int n) const noexcept -> int;

		int64_t const samplesPerPixel;
	};
}