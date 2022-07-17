module;
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
module Tracer.Spectrum:SampledSpectrum;
import Tracer.Spectrum;
import Math.Limits;
import Math.Common;
import :CoefficientSpectrum;
import :RGBSpectrum;
import :Common;

namespace SIByL::Tracer
{
    SampledSpectrum SampledSpectrum::X;
    SampledSpectrum SampledSpectrum::Y;
    SampledSpectrum SampledSpectrum::Z;
    SampledSpectrum SampledSpectrum::rgbRefl2SpectWhite;
    SampledSpectrum SampledSpectrum::rgbRefl2SpectCyan;
    SampledSpectrum SampledSpectrum::rgbRefl2SpectMagenta;
    SampledSpectrum SampledSpectrum::rgbRefl2SpectYellow;
    SampledSpectrum SampledSpectrum::rgbRefl2SpectRed;
    SampledSpectrum SampledSpectrum::rgbRefl2SpectGreen;
    SampledSpectrum SampledSpectrum::rgbRefl2SpectBlue;
    SampledSpectrum SampledSpectrum::rgbIllum2SpectWhite;
    SampledSpectrum SampledSpectrum::rgbIllum2SpectCyan;
    SampledSpectrum SampledSpectrum::rgbIllum2SpectMagenta;
    SampledSpectrum SampledSpectrum::rgbIllum2SpectYellow;
    SampledSpectrum SampledSpectrum::rgbIllum2SpectRed;
    SampledSpectrum SampledSpectrum::rgbIllum2SpectGreen;
    SampledSpectrum SampledSpectrum::rgbIllum2SpectBlue;

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
        auto interp = [&lambda, &v](float w, int i) {
            return std::lerp(v[i], v[i + 1],
                (w - lambda[i]) / (lambda[i + 1] - lambda[i]));
        };
        for (; i + 1 < n && lambda1 >= lambda[i]; ++i) {
            float segLambdaStart = std::max(lambda0, lambda[i]);
            float segLambdaEnd   = std::min(lambda1, lambda[i + 1]);
            sum += 0.5 * (interp(segLambdaStart, i) + interp(segLambdaEnd, i)) *
                (segLambdaEnd - segLambdaStart);
        }
        return sum / (lambda1 - lambda0);
    }

    SampledSpectrum::SampledSpectrum(RGBSpectrum const& s, SpectrumType type)
    {
        float rgb[3];
        s.toRGB(rgb);
        *this = SampledSpectrum::fromRGB(rgb, type);
    }

    auto SampledSpectrum::init() noexcept -> void
    {
        // Compute XYZ matching functions for SampledSpectrum
        for (int i = 0; i < nSpectralSamples; ++i) {
            float wl0 = std::lerp(sampledLambdaStart, sampledLambdaEnd, float(i) / float(nSpectralSamples));
            float wl1 = std::lerp(sampledLambdaStart, sampledLambdaEnd, float(i + 1) / float(nSpectralSamples));
            X.c[i] = AverageSpectrumSamples(CIE_lambda, CIE_X, nCIESamples, wl0, wl1);
            Y.c[i] = AverageSpectrumSamples(CIE_lambda, CIE_Y, nCIESamples, wl0, wl1);
            Z.c[i] = AverageSpectrumSamples(CIE_lambda, CIE_Z, nCIESamples, wl0, wl1);
        }
        // Compute RGB to spectrum functions for SampledSpectrum

    }

    auto SampledSpectrum::fromSampled(float const* lambda, float const* v, int n) noexcept -> SampledSpectrum
    {
        // Sort samples if unordered
        if (!SpectrumSamplesSorted(lambda, v, n)) {
            std::vector<float> slambda(&lambda[0], &lambda[n]);
            std::vector<float> sv(&v[0], &v[n]);
            SortSpectrumSamples(slambda.data(), sv.data(), n);
            return fromSampled(slambda.data(), sv.data(), n);
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
    
    auto SampledSpectrum::fromRGB(float const rgb[3], SpectrumType type) noexcept -> SampledSpectrum
    {
        SampledSpectrum r;
        if (type == SpectrumType::Reflectance) {
            // Convert reflectance spectrum to RGB
            if (rgb[0] <= rgb[1] && rgb[0] <= rgb[2]) {
                // Compute reflectance SampledSpectrum with rgb[0] as minimum
                r += rgb[0] * rgbRefl2SpectWhite;
                if (rgb[1] <= rgb[2]) {
                    r += (rgb[1] - rgb[0]) * rgbRefl2SpectCyan;
                    r += (rgb[2] - rgb[1]) * rgbRefl2SpectBlue;
                }
                else {
                    r += (rgb[2] - rgb[0]) * rgbRefl2SpectCyan;
                    r += (rgb[1] - rgb[2]) * rgbRefl2SpectGreen;
                }
            }
            else if (rgb[1] <= rgb[0] && rgb[1] <= rgb[2]) {
                // Compute reflectance SampledSpectrum with rgb[1] as minimum
                r += rgb[1] * rgbRefl2SpectWhite;
                if (rgb[0] <= rgb[2]) {
                    r += (rgb[0] - rgb[1]) * rgbRefl2SpectMagenta;
                    r += (rgb[2] - rgb[0]) * rgbRefl2SpectBlue;
                }
                else {
                    r += (rgb[2] - rgb[1]) * rgbRefl2SpectMagenta;
                    r += (rgb[0] - rgb[2]) * rgbRefl2SpectRed;
                }
            }
            else {
                // Compute reflectance SampledSpectrum with rgb[2] as minimum
                r += rgb[2] * rgbRefl2SpectWhite;
                if (rgb[0] <= rgb[1]) {
                    r += (rgb[0] - rgb[2]) * rgbRefl2SpectYellow;
                    r += (rgb[1] - rgb[0]) * rgbRefl2SpectGreen;
                }
                else {
                    r += (rgb[1] - rgb[2]) * rgbRefl2SpectYellow;
                    r += (rgb[0] - rgb[1]) * rgbRefl2SpectRed;
                }
            }
            r *= .94;
        }
        else {
            // Convert illuminant spectrum to RGB
            if (rgb[0] <= rgb[1] && rgb[0] <= rgb[2]) {
                // Compute illuminant SampledSpectrum with rgb[0] as minimum
                r += rgb[0] * rgbIllum2SpectWhite;
                if (rgb[1] <= rgb[2]) {
                    r += (rgb[1] - rgb[0]) * rgbIllum2SpectCyan;
                    r += (rgb[2] - rgb[1]) * rgbIllum2SpectBlue;
                }
                else {
                    r += (rgb[2] - rgb[0]) * rgbIllum2SpectCyan;
                    r += (rgb[1] - rgb[2]) * rgbIllum2SpectGreen;
                }
            }
            else if (rgb[1] <= rgb[0] && rgb[1] <= rgb[2]) {
                // Compute illuminant SampledSpectrum with rgb[1] as minimum
                r += rgb[1] * rgbIllum2SpectWhite;
                if (rgb[0] <= rgb[2]) {
                    r += (rgb[0] - rgb[1]) * rgbIllum2SpectMagenta;
                    r += (rgb[2] - rgb[0]) * rgbIllum2SpectBlue;
                }
                else {
                    r += (rgb[2] - rgb[1]) * rgbIllum2SpectMagenta;
                    r += (rgb[0] - rgb[2]) * rgbIllum2SpectRed;
                }
            }
            else {
                // Compute illuminant SampledSpectrum with rgb[2] as minimum
                r += rgb[2] * rgbIllum2SpectWhite;
                if (rgb[0] <= rgb[1]) {
                    r += (rgb[0] - rgb[2]) * rgbIllum2SpectYellow;
                    r += (rgb[1] - rgb[0]) * rgbIllum2SpectGreen;
                }
                else {
                    r += (rgb[1] - rgb[2]) * rgbIllum2SpectYellow;
                    r += (rgb[0] - rgb[1]) * rgbIllum2SpectRed;
                }
            }
            r *= .86445f;
        }
        return r.clamp();
    }
    
    auto SampledSpectrum::fromXYZ(float const xyz[3], SpectrumType type) noexcept -> SampledSpectrum
    {
        float rgb[3];
        XYZToRGB(xyz, rgb);
        return fromRGB(rgb, type);
    }

    auto SampledSpectrum::toXYZ(float xyz[3]) noexcept -> void
    {
        xyz[0] = xyz[1] = xyz[2] = 0.f;
        for (int i = 0; i < nSpectralSamples; ++i) {
            xyz[0] += X.c[i] * c[i];
            xyz[1] += Y.c[i] * c[i];
            xyz[2] += Z.c[i] * c[i];
        }
        float scale = float(sampledLambdaEnd - sampledLambdaStart) / float(CIE_Y_integral * nSpectralSamples);
        xyz[0] *= scale;
        xyz[1] *= scale;
        xyz[2] *= scale;
    }

    auto SampledSpectrum::y() noexcept -> float
    {
        float yy = 0.f;
        for (int i = 0; i < nSpectralSamples; ++i)
            yy += Y.c[i] * c[i];
        return yy * float(sampledLambdaEnd - sampledLambdaStart) / float(CIE_Y_integral * nSpectralSamples);
    }

    auto SampledSpectrum::toRGB(float rgb[3]) noexcept -> void
    {
        float xyz[3];
        toXYZ(xyz);
        XYZToRGB(xyz, rgb);
    }

    auto SampledSpectrum::toRGBSpectrum() const noexcept -> RGBSpectrum
    {
        return RGBSpectrum{};
    }
}