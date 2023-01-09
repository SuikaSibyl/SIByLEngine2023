#version 460
#extension GL_GOOGLE_include_directive : enable

#include "../include/common_trace.h"

layout(location = 0) rayPayloadInEXT PrimaryPayload primaryPld;

void main()
{
    primaryPld.baseColor = vec3(0.0);
    setIntersected(primaryPld.flags, false);
}