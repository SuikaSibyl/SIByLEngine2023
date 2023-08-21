#ifndef _SRENDERER_ADDON_PATHGUIDING_GAUSSIAN_EM_HEADER_
#define _SRENDERER_ADDON_PATHGUIDING_GAUSSIAN_EM_HEADER_

#include "../../include/common/cpp_compatible.hlsli"
#include "../../include/common/math.hlsli"
#include "../../include/common/gaussian.hlsli"

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



#endif // !_SRENDERER_ADDON_PATHGUIDING_GAUSSIAN_EM_HEADER_