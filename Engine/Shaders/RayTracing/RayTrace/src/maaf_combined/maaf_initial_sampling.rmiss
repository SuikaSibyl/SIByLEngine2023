#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "maaf_common.h"

layout(location = 0) rayPayloadInEXT PrimaryRayPayload pld;

void main() {
    pld.rayHitSky = true;
    pld.brdf = vec3(0);
}