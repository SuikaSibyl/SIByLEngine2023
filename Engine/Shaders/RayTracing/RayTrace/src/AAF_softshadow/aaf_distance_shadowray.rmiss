#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "aaf_payloads.h"

layout(location = 1) rayPayloadInEXT ShadowRayPayload shadowPayLoad;

void main() {
    // shadowPayLoad.hit = false;
}