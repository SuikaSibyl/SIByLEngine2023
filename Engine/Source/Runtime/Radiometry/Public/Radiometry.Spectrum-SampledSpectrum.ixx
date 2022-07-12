module;
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
export module Radiometry.Spectrum:SampledSpectrum;
import Math.Limits;
import Math.Common;
import :CoefficientSpectrum;

namespace SIByL::Radiometry
{
	inline constexpr size_t sampledLambdaStart = 400;
	inline constexpr size_t sampledLambdaEnd = 700;
	inline constexpr size_t nSpectralSamples = 60;

	export struct SampledSpectrum :public CoefficientSpectrum<nSpectralSamples>
	{
		SampledSpectrum(float v = 0.f) :CoefficientSpectrum(v) {}

		static auto Init() noexcept -> void;
		static auto FromSampled(float const* lambda, float const* v, int n) noexcept -> SampledSpectrum;

	};

	export inline int const nCIESamples = 471;
	extern float const CIE_X[nCIESamples];
	extern float const CIE_Y[nCIESamples];
	extern float const CIE_Z[nCIESamples];

	auto SpectrumSamplesSorted(float const* lambda, float const* v, int n) noexcept -> bool
	{
		for (int i = 0; i < n - 1; i++)
			if (lambda[i] > lambda[i + 1]) return false;
		return true;
	}

	auto SortSpectrumSamples(float* lambda, float* v, int n) noexcept -> void
	{
		std::vector<float> slambda(&lambda[0], &lambda[n]);
		std::vector<float> sv(&v[0], &v[n]);

		std::vector<std::size_t> permutation(slambda.size());
		std::iota(permutation.begin(), permutation.end(), 0);
		std::sort(permutation.begin(), permutation.end());

		std::transform(permutation.begin(), permutation.end(), &lambda[0],
			[&](std::size_t i) { return slambda[i]; });
		std::transform(permutation.begin(), permutation.end(), &v[0], 
			[&](std::size_t i) { return sv[i]; });
	}

	auto AverageSpectrumSamples(float const* lambda, float const* v, int n, float lambda0, float lambda1) noexcept -> float
	{
		// Handle cases with out-of-bounds range or single sample only
		if (lambda1 <= lambda[0])		return v[0];
		if (lambda0 >= lambda[n - 1])	return v[n - 1];
		if (n == 1)						return v[0];

		float sum = 0;
		// Add contributions of constant segments before/after samples
		if (lambda0 < lambda[0])
			sum += v[0] * (lambda[0] - lambda0);
		if (lambda1 > lambda[n - 1])
			sum += v[n - 1] * (lambda1 - lambda[n - 1]);
		// Advance to first relevant wavelent segments
		int i = 0;
		while (lambda0 > lambda[i + 1]) ++i;
		// Loop over wavelength segments and add contribitions
		auto interp = [lambda, v](float w, int i) {
			return std::lerp(v[i], v[i + 1], 
				(w - lambda[i]) / (lambda[i + 1] - lambda[i]));
		};
		for (; i + 1 < n && lambda1 >= lambda[i]; ++i) {
			float segLambdaStart = std::max(lambda0, lambda[i]);
			float segLambdaEnd = std::min(lambda1, lambda[i + 1]);
			sum += 0.5 * (interp(segLambdaStart, i) + interp(segLambdaEnd, i)) *
				(segLambdaEnd - segLambdaStart);
		}
		return sum / (lambda1 - lambda0);
	}
	
	auto SampledSpectrum::Init() noexcept -> void
	{
		// Compute XYZ matching functions for SampledSpectrum
		
		// Compute RGB to spectrum functions for SampledSpectrum

	}

	auto SampledSpectrum::FromSampled(float const* lambda, float const* v, int n) noexcept -> SampledSpectrum
	{
		if (!SpectrumSamplesSorted(lambda, v, n)) {
			std::vector<float> slambda(&lambda[0], &lambda[n]);
			std::vector<float> sv(&v[0], &v[n]);
			SortSpectrumSamples(slambda.data(), sv.data(), n);
			return FromSampled(slambda.data(), sv.data(), n);
		}
		
		SampledSpectrum r;
		for (int i = 0; i < nSpectralSamples; ++i) {
			// Compute average value of given SPD over i-th sample's range
			float lambda0 = std::lerp(sampledLambdaStart, sampledLambdaEnd, float(i) / float(nSpectralSamples));
			float lambda1 = std::lerp(sampledLambdaStart, sampledLambdaEnd, float(i + 1) / float(nSpectralSamples));
			r[i] = AverageSpectrumSamples(lambda, v, n, lambda0, lambda1);
		}
		return r;
	}
}