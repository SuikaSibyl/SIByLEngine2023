#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "maaf_common.h"

layout(location = 2) rayPayloadInEXT IndirectRayPayload indirectRayPayLoad;

void main() {
    indirectRayPayLoad.L = vec3(0.f);
    indirectRayPayLoad.hit = false;
}