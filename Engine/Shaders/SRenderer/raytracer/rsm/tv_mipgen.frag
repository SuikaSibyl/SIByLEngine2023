#version 460
#extension GL_GOOGLE_include_directive : enable
#include "../../../Utility/geometry.h"

layout(location = 0) in vec2 in_uv;

layout(location = 0) out float sum_visibility;

struct PushConstants { 
    ivec2 src_dim;
    ivec2 dst_dim;
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

layout(binding = 0) uniform sampler2D in_visibility;

void main() {
    const vec2 uv = in_uv;

    const ivec2 vWriteCoord = ivec2(uv * pushConstants.dst_dim);
    const ivec2 vReadCoord = vWriteCoord << 1;

    const float visibility_0 = texelFetch(in_visibility, vReadCoord + ivec2(0, 0), 0).x;
    const float visibility_1 = texelFetch(in_visibility, vReadCoord + ivec2(1, 0), 0).x;
    const float visibility_2 = texelFetch(in_visibility, vReadCoord + ivec2(0, 1), 0).x;
    const float visibility_3 = texelFetch(in_visibility, vReadCoord + ivec2(1, 1), 0).x;
    
    const float visibility = visibility_0 + visibility_1 + visibility_2 + visibility_3;    
	// Output
    sum_visibility = visibility;
}