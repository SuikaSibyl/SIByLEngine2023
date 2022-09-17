module;
#include <vector>
export module Tracer.Sampling:Distribution1D;

namespace SIByL::Tracer
{
	/**
	* A small utility class that represents a piecewise-constant 1D function's
	* PDF & CDF and provides methods to perform this sampling efficiency.
	*/
	struct Distribution1D
	{
		/**
		* @param f: 
		* @param n: the number of input constants f
		*/
		Distribution1D(float const* f, int n);

		auto count() const noexcept -> int { return func.size(); }
		auto sampleContinuous(float u, float* pdf, int* off = nullptr) const noexcept -> float;
		auto sampleDiscrete(float u, float* pdf = nullptr, float* uRemapped = nullptr) const noexcept -> int;
		
		/** Compute the PDF for sampling a given value from the discrete PDF */
		auto discretePDF(int index) const noexcept -> float;

		std::vector<float> func, cdf;
		float funcInt;
	};
}