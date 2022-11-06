#ifndef _SAMPLING_HEADER_
#define _SAMPLING_HEADER_

#include "math.h"

vec3 uniformSampleHemisphere(in vec2 u) {
    const float z = u.x;
    const float r = sqrt(max(0., 1. - z * z));
    const float phi = 2 * k_pi * u.y;
    return normalize(vec3(r * cos(phi), r * sin(phi), z));
}

vec3 uniformSampleHemisphere(in vec2 u, out float inv_pdf) {
    vec3 smp = uniformSampleHemisphere(u);
    inv_pdf = 2 * k_pi;
    return smp;
}

vec2 uniformSampleDisk(in vec2 u) {
    float r = sqrt(u.x);
    float theta = 2 * k_pi * u.y;
    return vec2(r * cos(theta), r * sin(theta));
}

vec2 concentricSampleDisk(in vec2 u) {
    // Map uniform random numbers to[−1, 1]^2
    vec2 uOffset = 2.f * u - vec2(1, 1);
    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0)
        return vec2(0, 0);
    // Apply concentric mapping to point
    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = k_pi_over_4 * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = k_pi_over_2 - k_pi_over_4 * (uOffset.x / uOffset.y);
    }
    return r * vec2(cos(theta), sin(theta));
}

vec3 cosineSampleHemisphere(in vec2 u) {
    vec2 d = concentricSampleDisk(u);
    float z = sqrt(max(0., 1 - d.x * d.x - d.y * d.y));
    return vec3(d.x, d.y, z);
}

vec3 cosineSampleHemisphere(in vec2 u, out float inv_pdf) {
    vec3 smp = cosineSampleHemisphere(u);
    inv_pdf = k_pi / abs(smp.z);
    return smp;
}

#endif
