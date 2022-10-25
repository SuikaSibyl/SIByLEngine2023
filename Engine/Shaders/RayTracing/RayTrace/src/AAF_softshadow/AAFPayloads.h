#ifndef _AAF_PAYLOADS_HEADER_
#define _AAF_PAYLOADS_HEADER_

struct ShadowRayPayload {
    bool hit;
    vec3 attenuation;
    float distanceMin;
    float distanceMax;
};

#endif