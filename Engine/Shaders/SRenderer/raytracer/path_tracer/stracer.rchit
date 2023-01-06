#version 460
#extension GL_GOOGLE_include_directive : enable

#include "../../include/common_hit.h"

layout(location = 0) rayPayloadInEXT vec3 hitValue;

void main()
{
    HitGeometry geoInfo = getHitGeometry();
    MaterialData material = materials[geoInfo.matID];
    vec3 base_color = texture(textures[material.basecolor_opacity_tex], geoInfo.uv).rgb;
    vec3 normal = texture(textures[material.normal_bump_tex], geoInfo.uv).rgb;
    normal = normalize(normal * 2.0 - 1.0);   
    normal = normalize(geoInfo.TBN * normal);
    hitValue = base_color;
}