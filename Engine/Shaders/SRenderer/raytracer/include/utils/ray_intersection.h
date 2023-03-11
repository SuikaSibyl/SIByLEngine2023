#ifndef _SRENDERER_COMMON_RT_RAY_INTERSECTION_HEADER_
#define _SRENDERER_COMMON_RT_RAY_INTERSECTION_HEADER_

#include "../../../../Utility/math.h"

/** A ray. */
struct Ray {
    vec3    origin;
    float   tMin;
    vec3    direction;
    float   tMax;
};

struct SurfaceIntersection {
    vec3 position;          // hit position
    uint geometryID;        // geometry ID
    vec3 geometric_normal;  // surface normal
    uint lightID;           // light ID
    vec3 wo;                // negative ray direction
    uint matID;             // material ID
    vec2 uv;                // uv for texture fetching
    float uv_screen_size;   // screen uv size
    float mean_curvature;   // for ray differential propagation

    float ray_radius;       // for ray differential propagation
    mat3 shading_frame;     // shading frame
    // vec3 offsetedPosition;
    // vec3 offsetedPositionInv;
    vec3 lightNormal;
    float hitFrontface;
};

/** Initialize a ray */
void initialize(
    inout Ray ray,
    in const vec3 origin,
    in const vec3 direction
) {
    ray.origin = origin;
    ray.direction = direction;
    ray.tMin = 0;
    ray.tMax = k_inf;
}

/** Initialize a ray */
void initialize(
    inout Ray ray,
    in const vec3 origin,
    in const vec3 direction,
    in const float tmin,
    in const float tmax
) {
    ray.origin = origin;
    ray.direction = direction;
    ray.tMin = tmin;
    ray.tMax = tmax;
}

// offsetPositionAlongNormal shifts a point on a triangle surface so that a
// ray bouncing off the surface with tMin = 0.0 is no longer treated as
// intersecting the surface it originated from.
//
// Here's the old implementation of it we used in earlier chapters:
// vec3 offsetPositionAlongNormal(vec3 worldPosition, vec3 normal)
// {
//   return worldPosition + 0.0001 * normal;
// }
//
// However, this code uses an improved technique by Carsten Wächter and
// Nikolaus Binder from "A Fast and Robust Method for Avoiding
// Self-Intersection" from Ray Tracing Gems (verion 1.7, 2020).
// The normal can be negated if one wants the ray to pass through
// the surface instead.

vec3 offsetPositionAlongNormal(vec3 worldPosition, vec3 normal) {
    // Convert the normal to an integer offset.
    const float int_scale = 256.0f;
    const ivec3 of_i      = ivec3(int_scale * normal);
    // Offset each component of worldPosition using its binary representation.
    // Handle the sign bits correctly.
    const vec3 p_i = vec3(  //
        intBitsToFloat(floatBitsToInt(worldPosition.x) + ((worldPosition.x < 0) ? -of_i.x : of_i.x)),
        intBitsToFloat(floatBitsToInt(worldPosition.y) + ((worldPosition.y < 0) ? -of_i.y : of_i.y)),
        intBitsToFloat(floatBitsToInt(worldPosition.z) + ((worldPosition.z < 0) ? -of_i.z : of_i.z)));
    // Use a floating-point offset instead for points near (0,0,0), the origin.
    const float origin     = 1.0f / 32.0f;
    const float floatScale = 1.0f / 65536.0f;
    return vec3(  //
        abs(worldPosition.x) < origin ? worldPosition.x + floatScale * normal.x : p_i.x,
        abs(worldPosition.y) < origin ? worldPosition.y + floatScale * normal.y : p_i.y,
        abs(worldPosition.z) < origin ? worldPosition.z + floatScale * normal.z : p_i.z);
}

Ray spawnRay(
    in const SurfaceIntersection isect, 
    in const vec3 dir
) {
    vec3 offsetDir = faceforward(isect.geometric_normal, -dir, isect.geometric_normal);
    vec3 offsetedPosition = offsetPositionAlongNormal(isect.position, offsetDir);
    
    Ray ray;
    ray.origin     = offsetedPosition;
    ray.direction  = dir;
    ray.tMin       = 0.000;
    ray.tMax       = k_inf;
    return ray;
}


#endif