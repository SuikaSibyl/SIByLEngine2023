module;
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
export module Tracer.Spectrum:SampledSpectrum;
import SE.Math.Misc;
import :CoefficientSpectrum;
import :Common;

namespace SIByL::Tracer
{
	export struct RGBSpectrum;

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
		SampledSpectrum(CoefficientSpectrum<nSpectralSamples> const& s) :CoefficientSpectrum(s) {}
		SampledSpectrum(RGBSpectrum const& s, SpectrumType type);

		/**
		* Create XYZ matching curves in SampledSpectrum representation
		* @see: static Members SampledSpectrum X/Y/Z
		*/
		static auto init() noexcept -> void;

		/**
		* Takes an arrays of {lambda,v} to define a piecewise linear function to present the SPD
		* At each SPD sample, compute the average of the function over the range
		*/
		static auto fromSampled(float const* lambda, float const* v, int n) noexcept -> SampledSpectrum;

		/** 
		* Convert from a given RGB to a full SPD
		* It also take an enumeration denoting whether RGB represents surface reflection or an illuminant
		* @ref Smits, Brian E.. “An RGB-to-Spectrum Conversion for Reflectances.” J. Graphics, GPU, & Game Tools 4 (1999): 11-22.
		*/
		static auto fromRGB(float const rgb[3], SpectrumType type) noexcept -> SampledSpectrum;

		/** Convert from a given XYZ to a full SPD */
		static auto fromXYZ(float const xyz[3], SpectrumType type) noexcept -> SampledSpectrum;

		/** Convert the SPD represents to XYZ spectral representations */
		auto toXYZ(float xyz[3]) noexcept -> void;

		/** Only calculate Y coeff of XYZ representation, wich is important cause closely realted to luminance */
		auto y() noexcept -> float;

		/** Convert the SPD represents to RGB spectral representations */
		auto toRGB(float rgb[3]) noexcept -> void;

		/** Convert the SPD represents to RGBSpectrum */
		auto toRGBSpectrum() const noexcept -> RGBSpectrum;

		/**
		* Static data created in Init(), SampledSpectrum representations of common SPD curves
		* @see SampledSpectrum::init()
		*/
		static SampledSpectrum X, Y, Z;
		static SampledSpectrum rgbRefl2SpectWhite, rgbRefl2SpectCyan;
		static SampledSpectrum rgbRefl2SpectMagenta, rgbRefl2SpectYellow;
		static SampledSpectrum rgbRefl2SpectRed, rgbRefl2SpectGreen;
		static SampledSpectrum rgbRefl2SpectBlue;
		static SampledSpectrum rgbIllum2SpectWhite, rgbIllum2SpectCyan;
		static SampledSpectrum rgbIllum2SpectMagenta, rgbIllum2SpectYellow;
		static SampledSpectrum rgbIllum2SpectRed, rgbIllum2SpectGreen;
		static SampledSpectrum rgbIllum2SpectBlue;
	};
}