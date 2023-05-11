#version 460
#extension GL_GOOGLE_include_directive : enable
#include "ssrgt_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload pld;

void main() {
    pld.hit = false;
}