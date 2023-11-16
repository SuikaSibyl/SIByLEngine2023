#ifndef SRENDERER_COMMON_RAYCON_HEADER
#define SRENDERER_COMMON_RAYCON_HEADER

#include "raycast.hlsli"

/**
 * Ray cone, a simple strategy to compute texture level of detail.
 * The strategies are introduces in Ray Tracing Gems I Chapter 20.
 * <Texture Level of Detail Strategies for Real-Time Ray Tracing>
 */
struct RayCone {
    float width;
    float spreadAngle;
};

/**
 * Propagates a ray cone given a surface spread angle and a hit distance.
 * @param cone: The cone to propagate.
 * @param surface_spread_angle: The surface spread angle to add to the cone.
 * @param hit_t: The distance at which the cone hit the surface.
 */
RayCone propagate(in_ref(RayCone) cone, float surface_spread_angle, float hit_t) {
    RayCone new_cone;
    new_cone.width = cone.spreadAngle * hit_t + cone.width;
    new_cone.spreadAngle = cone.spreadAngle + surface_spread_angle;
    return new_cone;
}

/**
 * Compute texture level of details based on ray cone.
 * @param lambda: primitive associcated parameter.
 * @param ray_direction: ray direction.
 * @param normal: normal at the hit point.
 * @return The level of detail, lambda.
 */
float computeLOD(
    in_ref(RayCone) cone,
    in_ref(float) lambda,
    in_ref(float3) ray_direction,
    in_ref(float3) normal
) {
    const float dist_term = abs(cone.width);
    const float normal_term = abs(dot(ray_direction, normal));
    lambda += log2(dist_term / normal_term);
    return lambda;
}

/**
 * Computes the texture level of detail for a given ray and cone.
 * @param ray: The ray to compute the texture level of detail for.
 * @param cone: The cone to compute the texture level of detail for.
 * @param lambda: The lambda value to use for the texture level of detail computation.
 * @param normal: The normal of the surface the ray hit.
 * @param texture_size: The size of the texture the ray is sampling.
 * @return The texture level of detail for the given ray and cone.
 */
float commputeTextureLoD(
    in_ref(Ray) ray, 
    in_ref(RayCone) cone,
    in_ref(float) lambda,
    in_ref(float3) normal,
    in_ref(int2) texure_size
) {
    lambda += log2(abs(cone.width));
    lambda += 0.5 * log2(texure_size.x * texure_size.y);
    lambda -= log2(abs(dot(ray.direction, normal)));
    return lambda;
}

/** Computes the pixel spread angle for a given camera. */
float pixelSpreadAngle(in_ref(CameraData) cameraData) {
    const int2 viewport = getViewportSize(cameraData);
    return atan(2.0 * length(cameraData.cameraV) / float(viewport.y));
}

#endif // SRENDERER_COMMON_RAYCON_HEADER