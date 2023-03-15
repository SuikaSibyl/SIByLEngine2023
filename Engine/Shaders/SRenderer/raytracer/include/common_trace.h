#ifndef _SRENDERER_COMMON_TRACE_HEADER_
#define _SRENDERER_COMMON_TRACE_HEADER_

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require

#include "common_rt_config.h"
#include "common_callable.h"
#include "../../include/common_descriptor_sets.h"
#include "../../include/common_utils.h"
#include "../../include/spectrum.h"

#include "utils/ray_intersection.h"
#include "utils/camera_model.h"

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
    float   hitFrontface; // ....
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

#endif