#version 460
#extension GL_GOOGLE_include_directive : enable

#include "../../include/common_descriptor_sets.h"
#include "../../include/plugins/material/principled_common.h"
#include "../../../Utility/geometry.h"

layout(location = 0) in vec2 uv;
layout(location = 1) in flat uint matID;
layout(location = 2) in vec3 normalWS;
layout(location = 3) in vec4 tangentWS;

layout(location = 0) out vec4 outColor;

void main() {
    mat3 TBN = buildTangentToWorld(tangentWS, normalWS);
    if(!gl_FrontFacing) TBN[2] = -TBN[2];

    // bump = normalize(dot(bump, TBN));
    // mat3 TBN = inTBN;
    // if(!gl_FrontFacing) TBN[2] = -TBN[2];

    PrincipledMaterialData material = principled_materials[matID];

    vec3 base_color = texture(textures[material.basecolor_opacity_tex], uv).rgb;
    // vec3 normal = texture(textures[material.normal_bump_tex], uv).rgb;
    // normal = normalize(normal * 2.0 - 1.0);   
    // normal = normalize(TBN * normal);

    outColor = vec4(base_color, 1.0);
}