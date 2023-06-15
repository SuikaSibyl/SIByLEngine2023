#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_fragment_shader_barycentric : require
#extension GL_EXT_scalar_block_layout : require

#include "../../include/common_descriptor_sets.h"
#include "../../include/plugins/material/principled_common.h"
#include "../../../Utility/geometry.h"

layout(location = 0) in vec2 uv;
layout(location = 1) in flat uint matID;
layout(location = 2) in vec3 normalWS;
layout(location = 3) in vec4 tangentWS;

layout(location = 0) out vec4 outColor;

struct GeoVisUniform {
    vec3  wireframe_color;
    uint  use_wireframe;
    float wireframe_smoothing;
    float wireframe_thickness;
};

layout(binding = 0, set = 1) uniform _GeoVisUniform { GeoVisUniform uni; };

vec3 GetAlbedoWithWireframe(in const vec3 color) {
	// vec3 albedo = GetAlbedo(i);
	vec3 barys = gl_BaryCoordEXT;
	const vec3 deltas = fwidth(barys);
	const vec3 smoothing = deltas * uni.wireframe_smoothing;
	const vec3 thickness = deltas * uni.wireframe_thickness;
	barys = smoothstep(thickness, thickness + smoothing, barys);
	float minBary = min(barys.x, min(barys.y, barys.z));
	return mix(uni.wireframe_color, color, minBary);
}

void main() {
    mat3 TBN = buildTangentToWorld(tangentWS, normalWS);
    if(!gl_FrontFacing) TBN[2] = -TBN[2];
    
    vec3 vbc = gl_BaryCoordEXT;

    // bump = normalize(dot(bump, TBN));
    // mat3 TBN = inTBN;
    // if(!gl_FrontFacing) TBN[2] = -TBN[2];

    PrincipledMaterialData material = principled_materials[matID];

    vec3 base_color = texture(textures[material.basecolor_opacity_tex], uv).rgb;
    // vec3 normal = texture(textures[material.normal_bump_tex], uv).rgb;
    // normal = normalize(normal * 2.0 - 1.0);   
    // normal = normalize(TBN * normal);
    vec3 output_color = base_color;

    if(uni.use_wireframe == 1u) {
        output_color = GetAlbedoWithWireframe(output_color);
    }

    outColor = vec4(output_color, 1.0);
}