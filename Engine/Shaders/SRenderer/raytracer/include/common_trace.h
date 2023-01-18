#ifndef _SRENDERER_COMMON_TRACE_HEADER_
#define _SRENDERER_COMMON_TRACE_HEADER_

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require

#include "common_rt_config.h"
#include "../../include/common_descriptor_sets.h"

layout(binding = 0, set = 1) uniform accelerationStructureEXT tlas;
layout(binding = 1, set = 1, rgba32f) uniform image2D storageImage;

/** Primary Payload Struct */
struct PrimaryPayload {
    vec3    position;       // 00: position of hit point
    uint    flags;          // 12: flags of the hit result
    vec2    uv;             // 16: uv of hit point
    uint    matID;          // 24: material ID of hit point
    uint    geometryID;     // 28: geometry ID of hit point
    vec3    geometryNormal; // 32: geometry normal
    uint    lightID;        // 44: light ID
    mat3    TBN;            // 48: TBN frame of hit point
    float   normalFlipping; // ....
};
/** Primary Payload Struct */
struct ShadowPayload {
    bool    occluded;
};

/** Flag Definition */
void setIntersected(inout uint flags, in bool intersected) { 
    flags = bitfieldInsert(flags, uint(intersected), 0, 1); }
bool getIntersected(in uint flags) {
    return bool(bitfieldExtract(flags, 0, 1)); }

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

#endif