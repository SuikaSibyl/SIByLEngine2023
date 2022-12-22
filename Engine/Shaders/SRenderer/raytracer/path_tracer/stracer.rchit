#version 460
#extension GL_GOOGLE_include_directive : enable

#include "../../include/common_hit.h"

layout(location = 0) rayPayloadInEXT vec3 hitValue;

void main()
{
    HitGeometry geoInfo = getHitGeometry();
    hitValue = vec3(geoInfo.uv, 0);
}