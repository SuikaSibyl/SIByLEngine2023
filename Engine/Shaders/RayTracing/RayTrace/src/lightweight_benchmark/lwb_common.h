#ifndef _LIGHTWEIGHT_BENCHMARK_HEADER_
#define _LIGHTWEIGHT_BENCHMARK_HEADER_

/** output image resolution */
const uvec2 resolution = uvec2(800, 600);

struct PrimaryRayPayload {
    vec3    hitPoint;       // position of hit point
    uint    rngState;       // State of the random number generator.
    vec3    hitNormal;      // normal of hit point
    vec3    color;
    
    bool    rayHitSky;      // True if the ray hit the sky.
    bool    rayHitReflector;   // True if one of shadow rays hit sth.
    bool    rayHitShadow;
};

vec3 skyColor(in vec3 rayDir) {
    const float t = 0.5 * (rayDir.y + 1.0);
    return (1.0-t)*vec3(1.0) + t*vec3(0.5, 0.7, 1.0);
}

#endif