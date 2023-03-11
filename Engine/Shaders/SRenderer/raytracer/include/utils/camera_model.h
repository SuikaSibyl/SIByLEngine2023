#ifndef _SRENDERER_COMMON_RT_CAMERA_MODEL_HEADER_
#define _SRENDERER_COMMON_RT_CAMERA_MODEL_HEADER_

#include "ray_intersection.h"

struct CameraSample {
    vec2 pFilm;
    vec2 pLens;
};

/** Randomly generate a camera ray. */
Ray generateRay(in const vec2 ndc) {
    Ray ray;
    ray.origin = globalUniform.cameraData.posW;
    // Compute the normalized ray direction assuming a pinhole camera.
    ray.direction = normalize(ndc.x * globalUniform.cameraData.cameraU 
        + ndc.y * globalUniform.cameraData.cameraV 
        + globalUniform.cameraData.cameraW);

    float invCos = 1.f / dot(normalize(globalUniform.cameraData.cameraW), ray.direction);
    ray.tMin = globalUniform.cameraData.nearZ * invCos * 2;
    ray.tMax = globalUniform.cameraData.farZ * invCos * 2;
    
    return ray;
}

/** Compute the ray corresponding to a given sample */
Ray generateRay(in const CameraSample cameraSample) {
    vec2 ndc = cameraSample.pFilm * 2 - vec2(1);
    return generateRay(ndc);
}

/** Compute the ray corresponding to a given pixel */
Ray generateRay(
    vec2 pixel,
    uvec2 frameDim,
    bool  applyJitter
) {
    // Compute the normalized ray direction assuming a pinhole camera.
    // Compute sample position in screen space in [0,1] with origin at the top-left corner.
    vec2 p = (vec2(pixel) + vec2(0.5f, 0.5f)) / vec2(frameDim);
    // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
    if (applyJitter)
        p += vec2(-globalUniform.cameraData.jitterX, globalUniform.cameraData.jitterY);
    // Compute ndc of the corresponding pixel.
    vec2 ndc = vec2(2.0f, -2.0f) * p + vec2(-1.0f, 1.0f);
    return generateRay(ndc);
}

#endif