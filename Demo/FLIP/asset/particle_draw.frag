#version 460
#extension GL_EXT_fragment_shader_barycentric : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : enable
#include "particle_draw.glsl"

layout(location = 0) in vec3 col;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(col, 1.0);
}
