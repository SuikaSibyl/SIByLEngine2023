/* **************************************************
 * Copyright (c) 2023, Haolin Lu <hal128@ucsd.edu>
 * All rights reserved.
 ***************************************************/

#ifndef _SRENDERER_GAUSSIAN_HEADER_
#define _SRENDERER_GAUSSIAN_HEADER_

#include "cpp_compatible.hlsli"
#include "math.hlsli"
#include "linear_algebra.hlsli"

/**
 * Box-Muller transform takes two uniform samples u0 and u1,
 * and transforms them into two Gaussian distributed samples n0 and n1.
 * @param uv 2 uniform samples in range [0, 1]
 * @return 2 standard Gaussian distributed samples.
 */
float2 BoxMuller(in_ref(float2) uv) {
    const float radius = sqrt(-2.0 * log(uv.x));
    const float theta = 2 * k_pi * uv.y;
    return float2(radius * cos(theta), radius * sin(theta));
}

struct NormalDistribution {
    float2 mean;         // mean of the distribution
    float det;           // determinant of the covariance matrix
    float2x2 covariance; // covariance matrix
    float2x2 inverse;    // inverse of the covariance matrix
    float2x2 cholesky;   // Cholesky decomposition of the covariance matrix
    // functions
    // ---------------------------------------------------------------------------------
    // constructors
    __init(float2 mean, float2x2 covariance) {
        det = covariance._11 * covariance._22 - covariance._12 * covariance._21;
        inverse = Inverse2x2(covariance, det);
        cholesky = CholeskyDecomposition2x2(covariance);
    }
    // Draw a sample from the distribution
    float2 DrawSample(in_ref(float2) uv) {
        // Generate two independent random variables from a standard normal distribution
        const float2 n = BoxMuller(uv);
        // Transform the independent variables to match the desired mean and covariance
        return mean + mul(n, cholesky);
    }
    // evaluate the probability density function at the given sample
    float Pdf(in_ref(float2) sample) {
        const float2 x = sample - mean;
        return (1.0f / (k_2pi * sqrt(det))) * exp(-0.5f * dot(x, mul(inverse, x)));
    }
};

#endif // !_SRENDERER_GAUSSIAN_HEADER_