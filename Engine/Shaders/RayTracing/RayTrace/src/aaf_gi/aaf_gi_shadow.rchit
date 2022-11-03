#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "aaf_gi_common.h"

layout(location = 2) rayPayloadInEXT ShadowRayPayload shadowRayPayLoad;

void main() {
    shadowRayPayLoad.hit = true;
}