#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : require
#include "camera_def.glsl"

layout(binding = 0, set = 0, scalar) uniform _GlobalUniforms  { CameraData gCamera; };
layout(binding = 1, set = 0) uniform sampler2D in_color;

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 outColor;

void main() {
    const vec4 color = texture(in_color, in_uv);
    outColor = color;
}
