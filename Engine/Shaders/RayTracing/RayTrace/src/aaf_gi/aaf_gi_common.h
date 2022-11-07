#ifndef _AAF_GI_COMMON_HEADER_
#define _AAF_GI_COMMON_HEADER_

#include "../../../../Utility/math.h"

/** output image resolution */
const uvec2 resolution = uvec2(800, 600);
/* light position */
const vec3 lightPos = vec3(-0.24 + 0.47/2, 1.979, -0.22 + 0.38/2);

const vec3 Kd = vec3(0.87402f, 0.87402f, 0.87402f);

struct PrimarySamplePayload {
    vec3    hitPoint;       // position of hit point
    uint    rngState;       // State of the random number generator.
    vec3    hitNormal;      // normal of hit point
    float   slope1;
    vec3    accumGI;        // accumulated weighted GI
    float   accumWeights;   // accumulated weights
    vec3    brdf;           // brdf of the hit point
    float   slope2;
    bool    rayHitSky;      // True if the ray hit the sky.
    bool    rayHitReflector;   // True if one of shadow rays hit sth.
    bool    rayHitShadow;
};

struct SecondaryRayPayload {
    vec3    L;
    float   distanceToReflector;
    bool    hit;
};

struct ShadowRayPayload {
    bool hit;
};

vec3 getAlbedo(vec3 worldNormal) {
    vec3 color = vec3(0.8f);
    const float dotX = dot(worldNormal, vec3(1.0, 0.0, 0.0));
    if(dotX > 0.99)
        color = vec3(0.8, 0.0, 0.0);
    else if(dotX < -0.99)
        color = vec3(0.0, 0.8, 0.0);
    return color;
}

#endif