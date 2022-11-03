#ifndef _AAF_PAYLOADS_HEADER_
#define _AAF_PAYLOADS_HEADER_

struct InitalSamplePayload {
    vec3 hitPoint;          // position of hit point
    float accumVis;         // accumulated weighted visibility
    vec3 hitNormal;         // normal of hit point
    float accumWeights;     // accumulated weights

    uint rngState;      // State of the random number generator.
    bool rayHitSky;     // True if the ray hit the sky.
    bool rayShadow;     // True if one of shadow rays hit sth.
    float s1;
    float s2;
    vec3 brdf;         // brdf of the hit point
};

struct ShadowRayPayload {
    bool hit;
    vec3 attenuation;
    float distanceMin;
    float distanceMax;
};

#endif