#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "../include/common_trace.h"

layout(location = 1) rayPayloadInEXT ShadowPayload rShadowPld;

void main() {
    rShadowPld.occluded = false;
}