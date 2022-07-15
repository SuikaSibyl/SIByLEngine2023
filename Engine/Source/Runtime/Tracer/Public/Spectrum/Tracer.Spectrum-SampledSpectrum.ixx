module;
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
export module Tracer.Spectrum:SampledSpectrum;
import Math.Limits;
import Math.Common;
import :CoefficientSpectrum;

namespace SIByL::Tracer
{
	inline constexpr size_t sampledLambdaStart = 400;
	inline constexpr size_t sampledLambdaEnd = 700;
	inline constexpr size_t nSpectralSamples = 60;

	/**
	* Sampled Spectrum represent an SPD with uniformly spaced samples
	* between wavelength [400-700] nm, with 60 samples.
	*/
	export struct SampledSpectrum :public CoefficientSpectrum<nSpectralSamples>
	{
		SampledSpectrum() :CoefficientSpectrum(0.f) {}
		SampledSpectrum(float v) :CoefficientSpectrum(v) {}

		static auto Init() noexcept -> void;

		/**
		* Takes an arrays of {lambda,v} to define a piecewise linear function to present the SPD
		* At each SPD sample, compute the average of the function over the range
		*/
		static auto FromSampled(float const* lambda, float const* v, int n) noexcept -> SampledSpectrum;



		/** Convert the SPD represents to RGB spectral representations */


		/** Convert the SPD represents to XYZ spectral representations */
	};
}