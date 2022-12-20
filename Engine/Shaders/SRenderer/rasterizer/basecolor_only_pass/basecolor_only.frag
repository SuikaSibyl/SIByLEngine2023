#version 460
// #extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec2 uv;
layout(location = 1) in flat uint matID;

layout(location = 0) out vec4 outColor;

// layout(binding = 1) uniform sampler2D textures[];

void main() {
    outColor = vec4(uv, 0, 1.0);
}
