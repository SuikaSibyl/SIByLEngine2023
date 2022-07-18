module;
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
module Tracer.Spectrum:RGBSpectrum;
import Tracer.Spectrum;
import :SampledSpectrum;
import :CoefficientSpectrum;
import :Common;
import Math.Limits;
import Math.Common;

namespace SIByL::Tracer
{
	auto RGBSpectrum::fromRGB(float const rgb[3], SpectrumType type) noexcept -> RGBSpectrum {
		RGBSpectrum s;
		s.c[0] = rgb[0];
		s.c[1] = rgb[1];
		s.c[2] = rgb[2];
		return s;
	}
	
	auto RGBSpectrum::fromXYZ(float const xyz[3], SpectrumType type) noexcept -> RGBSpectrum {
		RGBSpectrum r;
		XYZToRGB(xyz, r.c);
		return r;
	}
	
	auto RGBSpectrum::fromSampled(float const* lambda, float const* v, int n) noexcept -> RGBSpectrum {
		// Sort samples if unordered, use sorted for returned spectrum
		if (!spectrumSamplesSorted(lambda, v, n)) {
			std::vector<float> slambda(&lambda[0], &lambda[n]);
			std::vector<float> sv(&v[0], &v[n]);
			sortSpectrumSamples(slambda.data(), sv.data(), n);
			return fromSampled(slambda.data(), sv.data(), n);
		}

		float xyz[3] = { 0,0,0 };
		for (int i = 0; i < nCIESamples; ++i) {
			float val = interpolateSpectrumSamples(lambda, v, n, CIE_lambda[i]);
			xyz[0] += val * CIE_X[i];
			xyz[1] += val * CIE_Y[i];
			xyz[2] += val * CIE_Z[i];
		}
		float scale = float(CIE_lambda[nCIESamples - 1] - CIE_lambda[0]) /
			float(CIE_Y_integral * nCIESamples);
		xyz[0] *= scale;
		xyz[1] *= scale;
		xyz[2] *= scale;
		return fromXYZ(xyz);
	}

	auto RGBSpectrum::toRGB(float rgb[3]) const noexcept -> void {
		rgb[0] = c[0];
		rgb[1] = c[1];
		rgb[2] = c[2];
	}

	auto RGBSpectrum::toXYZ(float xyz[3]) const noexcept -> void {
		RGBToXYZ(c, xyz);
	}

	auto RGBSpectrum::toRGBSpectrum() const noexcept -> RGBSpectrum {
		return *this;
	}

	auto RGBSpectrum::y() const noexcept -> float {
		float const YWeight[3] = { 0.212671f, 0.715160f, 0.072169f };
		return YWeight[0] * c[0] + YWeight[1] * c[1] + YWeight[2] * c[2];
	}

}