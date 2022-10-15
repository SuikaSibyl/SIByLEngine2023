
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

vec3 randomPointInSphere(inout uint rngState) {
    const float theta = 6.2831853 * stepAndOutputRNGFloat(rngState);   // Random in [0, 2pi]
    const float u     = 2.0 * stepAndOutputRNGFloat(rngState) - 1.0;  // Random in [-1, 1]
    const float r     = sqrt(1.0 - u * u);
    return vec3(r * cos(theta), r * sin(theta), u);
}