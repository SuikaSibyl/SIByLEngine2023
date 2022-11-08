#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "aaf_gi_common.h"

layout(location = 1) rayPayloadInEXT SecondaryRayPayload secondaryPayLoad;

void main() {
    secondaryPayLoad.L = vec3(0.f);
    secondaryPayLoad.hit = false;
}