#ifndef _RANDOM_HEADER_
#define _RANDOM_HEADER_

#include "math.h"

/**
* Random Number generator
* pcg32i_random_t based.
* @ref: https://www.pcg-random.org/index.html
* @ref: https://nvpro-samples.github.io/vk_mini_path_tracer/index.html#perfectlyspecularreflections/refactoringintersectioninformation
*/

// Random number generation using pcg32i_random_t, using inc = 1. 
// Our random state is a uint.
uint stepRNG(uint rngState) {
  return rngState * 747796405 + 1;
}

// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
float stepAndOutputRNGFloat(inout uint rngState) {
    // Condensed version of pcg_output_rxs_m_xs_32_32, 
    // with simple conversion to floating-point [0,1].
    rngState  = stepRNG(rngState);
    uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
    word      = (word >> 22) ^ word;
    return float(word) / 4294967295.0f;
}

float UniformFloat(inout uint rngState) {
  return stepAndOutputRNGFloat(rngState);
}

/**
*  Non-Uniform Sampling
*/

// Uses the Box-Muller transform to return a normally distributed (centered
// at 0, standard deviation 1) 2D point.
vec2 randomGaussian(inout uint rngState) {
  // Almost uniform in (0,1] - make sure the value is never 0:
  const float u1    = max(1e-38, stepAndOutputRNGFloat(rngState));
  const float u2    = stepAndOutputRNGFloat(rngState);  // In [0, 1]
  const float r     = sqrt(-2.0 * log(u1));
  const float theta = 2 * k_pi * u2;  // Random in [0, 2pi]
  return r * vec2(cos(theta), sin(theta));
}

/**
* Uniform Random in 3D geometics
*/

vec3 randomPointInSphere(inout uint rngState) {
    const float theta = 2 * k_pi * stepAndOutputRNGFloat(rngState);   // Random in [0, 2pi]
    const float u     = 2.0 * stepAndOutputRNGFloat(rngState) - 1.0;  // Random in [-1, 1]
    const float r     = sqrt(1.0 - u * u);
    return vec3(r * cos(theta), r * sin(theta), u);
}

vec3 randomPointInSphere(in const vec2 rvec) {
    const float theta = 2 * k_pi * rvec.x;   // Random in [0, 2pi]
    const float u     = 2.0 * rvec.y - 1.0;  // Random in [-1, 1]
    const float r     = sqrt(1.0 - u * u);
    return vec3(r * cos(theta), r * sin(theta), u);
}

vec4 uniformGenVec4(inout uint RNG) {
    return vec4(
      stepAndOutputRNGFloat(RNG),
      stepAndOutputRNGFloat(RNG),
      stepAndOutputRNGFloat(RNG),
      stepAndOutputRNGFloat(RNG)
    );
}

#endif