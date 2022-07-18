module;
export module Tracer.Spectrum:Common;

namespace SIByL::Tracer
{
    /**
    * Convert RGB to Spectrum will use different coefficients
    * when RGB represents surface reflection or an illuminaion
    */
    export enum struct SpectrumType {
        Reflectance,
        Illuminant
    };

    export inline int constexpr    nCIESamples = 471;
    export inline float constexpr  CIE_Y_integral = 106.856895;

    export extern float const CIE_X[nCIESamples];
    export extern float const CIE_Y[nCIESamples];
    export extern float const CIE_Z[nCIESamples];
    export extern float const CIE_lambda[nCIESamples];

    // RGB -> Full Spectrum
    export inline int constexpr nRGB2SpectSamples = 32;
    export extern float const RGB2SpectLambda[nRGB2SpectSamples];
    export extern float const RGBRefl2SpectWhite[nRGB2SpectSamples];
    export extern float const RGBRefl2SpectCyan[nRGB2SpectSamples];
    export extern float const RGBRefl2SpectMagenta[nRGB2SpectSamples];
    export extern float const RGBRefl2SpectYellow[nRGB2SpectSamples];
    export extern float const RGBRefl2SpectRed[nRGB2SpectSamples];
    export extern float const RGBRefl2SpectGreen[nRGB2SpectSamples];
    export extern float const RGBRefl2SpectBlue[nRGB2SpectSamples];
    export extern float const RGBIllum2SpectWhite[nRGB2SpectSamples];
    export extern float const RGBIllum2SpectCyan[nRGB2SpectSamples];
    export extern float const RGBIllum2SpectMagenta[nRGB2SpectSamples];
    export extern float const RGBIllum2SpectYellow[nRGB2SpectSamples];
    export extern float const RGBIllum2SpectRed[nRGB2SpectSamples];
    export extern float const RGBIllum2SpectGreen[nRGB2SpectSamples];
    export extern float const RGBIllum2SpectBlue[nRGB2SpectSamples];

    export inline auto XYZToRGB(float const xyz[3], float rgb[3]) noexcept -> void;
    export inline auto RGBToXYZ(float const rgb[3], float xyz[3]) noexcept -> void;

    /**
    * Takes a possibly irregularly sampled set of wavelengths
    * and SPD values (λi, vi) and returns the value of the SPD at the given wavelength λ,
    * linearly interpolating between the two sample values that bracket λ
    */
    export inline auto interpolateSpectrumSamples(const float* lambda, const float* vals, int n, float l) noexcept -> float;

    export inline auto spectrumSamplesSorted(float const* lambda, float const* v, int n) noexcept -> bool;

    export inline auto sortSpectrumSamples(float* lambda, float* v, int n) noexcept -> void;

    export inline auto averageSpectrumSamples(float const* lambda, float const* v, int n, float lambda0, float lambda1) noexcept -> float;
}