#ifndef SRENDERER_COMMON_RAYDIFFERENTIAL_HEADER
#define SRENDERER_COMMON_RAYDIFFERENTIAL_HEADER

#include "raycast.hlsli"

/** Ray differential structure for texture LoD level determination. */
struct RayDifferential {
    float3 dodx;
    float3 dody;
    float3 dddx;
    float3 dddy;
};

/** Constructor of ray differential. */
RayDifferential RayDifferential_ctor(
    in const float3 dodx,
    in const float3 dody,
    in const float3 dddx,
    in const float3 dddy
) {
    RayDifferential raydiff;
    raydiff.dodx = dodx;
    raydiff.dody = dody;
    raydiff.dddx = dddx;
    raydiff.dddy = dddy;
    return raydiff;
}

/**
 * If more than 1spp is used per pixel, the actual distance between samples is lower,
 * and the ray differentials need to be scaled accordingly.
 * @param ray The primary ray assoicated.
 * @param raydiff The ray differentials to scale.
 * @param s The scale factor standing for the sample spacing.
 */
void scale_differentials(in_ref(Ray) ray, inout_ref(RayDifferential) raydiff, float s) {
    raydiff.dodx = ray.origin + (raydiff.dodx - ray.origin) * s;
    raydiff.dody = ray.origin + (raydiff.dody - ray.origin) * s;
    raydiff.dddx = ray.direction + (raydiff.dddx - ray.direction) * s;
    raydiff.dddy = ray.direction + (raydiff.dddy - ray.direction) * s;
}

/**
 * Propagate the ray differential t distances away.
 * @param o Ray origin.
 * @param d Ray direction.
 * @param t The distance to the hit point.
 * @param n The normal at the hit point.
 * @return The propagated ray differential.
 */
RayDifferential propagate(in_ref(RayDifferential) raydiff, float3 O, float3 D, float t, float3 N) {
    float3 dodx = raydiff.dodx + t * raydiff.dddx;
    float3 dody = raydiff.dody + t * raydiff.dddy;
    const float rcpDN = 1.0f / dot(D, N);
    const float dtdx = -dot(dodx, N) * rcpDN;
    const float dtdy = -dot(dody, N) * rcpDN;
    dodx += D * dtdx;
    dody += D * dtdy;
    return RayDifferential_ctor(dodx, dody, raydiff.dddx, raydiff.dddy);
}

/** Compute the ray corresponding to a given pixel */
RayDifferential generateRayDifferential(
    float2 pixel,
    uint2 frameDim,
    bool applyJitter,
    in_ref(CameraData) cameraData
) {
    RayDifferential raydiff;
    Ray rx = generateRay(pixel + float2(1, 0), frameDim, applyJitter, cameraData);
    Ray ry = generateRay(pixel + float2(0, 1), frameDim, applyJitter, cameraData);
    raydiff.dodx = rx.origin;
    raydiff.dody = ry.origin;
    raydiff.dddx = rx.direction;
    raydiff.dddy = ry.direction;
    return raydiff;
}

#endif // !SRENDERER_COMMON_RAYDIFFERENTIAL_HEADER