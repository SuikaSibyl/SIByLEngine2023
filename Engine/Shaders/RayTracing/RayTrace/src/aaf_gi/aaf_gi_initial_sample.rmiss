#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "aaf_gi_common.h"

layout(location = 0) rayPayloadInEXT PrimarySamplePayload pld;

void main() {
    pld.rayHitSky = true;
}