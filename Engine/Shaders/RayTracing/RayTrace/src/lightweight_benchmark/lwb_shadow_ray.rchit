#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "lwb_common.h"

layout(location = 1) rayPayloadInEXT bool hitOccluder;

void main() {
    hitOccluder = true;
}