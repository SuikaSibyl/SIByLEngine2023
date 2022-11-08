#ifndef _AAF_GI_COMMON_HEADER_
#define _AAF_GI_COMMON_HEADER_

#include "../../../../Utility/math.h"

/** output image resolution */
const uvec2 resolution = uvec2(800, 600);
/* light position */
const vec3 lightPos = vec3(-0.24 + 0.47/2, 1.979, -0.22 + 0.38/2);
/* half the fov */
const float half_fov = 30.0;
/** pixel radius */
const ivec2 pixel_radius = ivec2(10, 10);
const float dist_scale_threshold = 10.0;
const float dist_threshold = 0.1;
const float angle_threshold = 20.0 * k_pi /180.0;

const float mu = 0.9f;
const float alpha = 0.3f;
const vec3 Kd = vec3(1);

const float maximumSceneDim = 2.f;
const float zMIN = 0.02 * maximumSceneDim;
const float zMAXMIN = 10 * zMIN;
const float omegaHMax = 2.8;
const float gamma = 0.4;

struct PrimarySamplePayload {
    vec3    hitPoint;
    uint    rngState;
    vec3    hitNormal;
    float   z_min;
    vec3    albedo;
    float   z_max;
    vec3    indirect;
    vec3    direct;
    bool    rayHitSky;      // True if the ray hit the sky.
    bool    rayHitReflector;   // True if one of shadow rays hit sth.
    bool    rayHitShadow;
};

struct SecondaryRayPayload {
    vec3    L;
    float   distanceToReflector;
    vec3    worldPosition;
    vec3    worldNormal;
    vec3    albedo;
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

float computeOmegaXMax(in float proj_dist) {
    return alpha / proj_dist;
}

float computeSPP(in float zmin, in float zmax, in float proj_dist) {
    const float sqrtAp = proj_dist;
    const float term1 = mu * omegaHMax * sqrtAp / zmin + alpha;
    const float term2 = omegaHMax;
    const float term3 = 1 + mu * zmax / zmin;
    return gamma * term1*term1 * term2*term2 * term3*term3;
}

float computeBeta(in float zmin, in float proj_dist) {
    const float omegaXMax = computeOmegaXMax(proj_dist);
    const float omegaXR = mu * min(omegaHMax/zmin, omegaXMax);
    return 2 / omegaXR; //beta = 2/omega_x^r
}

float gaussian(float distsq, float beta) {
    const float sqrt_2_pi = sqrt(2*k_pi);
    const float exponent = - distsq / (2 * beta * beta);
    return exp(exponent) / (sqrt_2_pi * beta);
}

#endif