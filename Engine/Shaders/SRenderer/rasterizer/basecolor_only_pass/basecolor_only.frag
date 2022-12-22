#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "../../include/common_descriptor_sets.h"

layout(location = 0) in vec2 uv;
layout(location = 1) in flat uint matID;
layout(location = 2) in vec3 color;
layout(location = 3) in mat3 TBN;

layout(location = 0) out vec4 outColor;

void main() {
    MaterialData material = materials[matID];

    vec3 base_color = texture(textures[material.basecolor_opacity_tex], uv).rgb;
    vec3 normal = texture(textures[material.normal_bump_tex], uv).rgb;
    // normal = normalize(normal * 2.0 - 1.0);   
    // normal = normalize(fs_in.TBN * normal);

    outColor = vec4(normal, 1.0);
}