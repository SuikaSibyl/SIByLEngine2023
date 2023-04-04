#version 460
#extension GL_GOOGLE_include_directive : enable

#include "../include/common_trace.h"

layout(location = 0) rayPayloadInEXT PrimaryPayload primaryPld;

void main()
{
    setIntersected(primaryPld.flags, false);
}