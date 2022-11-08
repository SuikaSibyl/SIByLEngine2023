#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "lwb_common.h"

layout(location = 0) rayPayloadInEXT PrimaryRayPayload pld;

void main() {
    pld.rayHitSky = true;
#if BENCHMARK == 1
    pld.color = skyColor(gl_WorldRayDirectionEXT);
#elif BENCHMARK == 2
    pld.color = vec3(0.f);
#endif
}