#ifndef _SAMPLING_HEADER_
#define _SAMPLING_HEADER_

#include "math.h"
	
vec3 uniformSampleHemisphere(in vec2 u) {
    const float z = u.x;
    const float r = sqrt(max(0., 1. - z * z));
    const float phi = 2 * k_pi * u.y;
    return normalize(vec3(r * cos(phi), r * sin(phi), z));
}

#endif
