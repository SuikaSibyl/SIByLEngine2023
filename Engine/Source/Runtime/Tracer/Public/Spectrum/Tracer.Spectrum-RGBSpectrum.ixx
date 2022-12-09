module;
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits>
export module Tracer.Spectrum:RGBSpectrum;
import :CoefficientSpectrum;
import :Common;
import SE.Math.Misc;

namespace SIByL::Tracer
{
	/**
	* Represent SPDs with a weighted sum of red, green & blue components
	*/
	export struct RGBSpectrum :public CoefficientSpectrum<3>
	{
		RGBSpectrum() :CoefficientSpectrum<3>(0) {}
		RGBSpectrum(float v) :CoefficientSpectrum<3>(v) {}
		RGBSpectrum(CoefficientSpectrum<3> const& v) :CoefficientSpectrum<3>(v) {}

		/**
		* Convert from a given RGB to a RGB Spectrum representation
		* It also take an enumeration denoting whether RGB represents surface reflection or an illuminant
		* which is only used for keep the interface similar
		* @see SampledSpectrum::fromRGB
		*/
		static auto fromRGB(float const rgb[3], SpectrumType type = SpectrumType::Reflectance) noexcept -> RGBSpectrum;
		/** Convert from a given XYZ to a RGB Spectrum representation */
		static auto fromXYZ(float const xyz[3], SpectrumType type = SpectrumType::Reflectance) noexcept -> RGBSpectrum;
		/** Convert from a given set of samples to a RGB Spectrum representation */
		static auto fromSampled(float const* lambda, float const* v, int n) noexcept -> RGBSpectrum;

		/** Convert specturm to RGB. Exist for satisfying common interface of Spectrum */
		auto toRGB(float rgb[3]) const noexcept -> void;
		/** Convert specturm to XYZ */
		auto toXYZ(float xyz[3]) const noexcept -> void;
		/** Convert specturm to Y of XYZ, which is related to illuminance */
		auto y() const noexcept -> float;
		/** Convert specturm to RGBSpectrum. Exist for satisfying common interface of Spectrum */
		auto toRGBSpectrum() const noexcept -> RGBSpectrum;
	};

	export using Spectrum = RGBSpectrum;
}