#version 460
#extension GL_EXT_ray_tracing : require

#include "AAFPayloads.h"

layout(location = 1) rayPayloadInEXT ShadowRayPayload shadowPayLoad;

void main() {
    shadowPayLoad.hit           = true;
    shadowPayLoad.attenuation   = vec3(0.f);
    shadowPayLoad.distanceMin   = vec3(0.f);
    shadowPayLoad.distanceMax   = vec3(0.f);
}