#ifndef _SRENDERER_ADDON_PATHGUIDING_GAUSSIAN_EM_HEADER_
#define _SRENDERER_ADDON_PATHGUIDING_GAUSSIAN_EM_HEADER_

#include "../../../include/common/cpp_compatible.hlsli"
#include "../../../include/common/gaussian.hlsli"
#include "../../../include/common/math.hlsli"

/**
 * Compute the covariance matrix of a 2D variable pair.
 * @param ex The expected value of the first variable.
 * @param ey The expected value of the second variable.
 * @param ex2 The expected value of the first variable squared.
 * @param ey2 The expected value of the second variable squared.
 * @param exy The expected value of the product of the two variables.
 * @return The covariance matrix.
*/
float2x2 Covariance2x2(float ex, float ey, float ex2, float ey2, float exy) {
    const float vx = ex2 - ex * ex;  // variance of x
    const float vy = ey2 - ey * ey;  // variance of y
    const float vxy = exy - ex * ey; // covariance of x and y
    return float2x2(vx, vxy, vxy, vy);
}

struct GMMStatictics {
    float ex;
    float ey;
    float ex2;
    float ey2;
    float exy;
};

GMMStatictics UnpackStatistics(in_ref(float4) pack0, in_ref(float4) pack1) {
    const float weightSum = pack1.y;
    GMMStatictics stat;
    stat.ex = pack0.x / weightSum;
    stat.ey = pack0.y / weightSum;
    stat.ex2 = pack0.z / weightSum;
    stat.ey2 = pack0.w / weightSum;
    stat.exy = pack1.x / weightSum;
    return stat;
}

float2x2 Covariance2x2(in_ref(GMMStatictics) statistics) {
    return Covariance2x2(statistics.ex, statistics.ey, statistics.ex2, statistics.ey2, statistics.exy);
}

MultivariateGaussian2D CreateDistribution(in_ref(GMMStatictics) statistics) {
    float2x2 covariance = Covariance2x2(statistics);
    return MultivariateGaussian2D(float2(statistics.ex, statistics.ey), covariance);
}

#endif // !_SRENDERER_ADDON_PATHGUIDING_GAUSSIAN_EM_HEADER_