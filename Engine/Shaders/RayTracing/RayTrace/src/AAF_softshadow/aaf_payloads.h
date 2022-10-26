#ifndef _AAF_PAYLOADS_HEADER_
#define _AAF_PAYLOADS_HEADER_

struct InitalSamplePayload {
    uint rngState;      // State of the random number generator.
    bool rayHitSky;     // True if the ray hit the sky.
    bool rayShadow;     // True if one of shadow rays hit sth.
    float distanceMin;
    float distanceMax;
};

struct ShadowRayPayload {
    bool hit;
    vec3 attenuation;
    float distanceMin;
    float distanceMax;
};

#endif