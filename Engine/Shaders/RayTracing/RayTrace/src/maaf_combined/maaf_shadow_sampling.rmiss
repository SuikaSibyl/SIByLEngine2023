#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "maaf_common.h"

layout(location = 3) rayPayloadInEXT   bool shadowHit;

void main() {
    shadowHit = false;
}