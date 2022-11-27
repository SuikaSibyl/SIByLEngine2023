#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D textures[];

void main() {
    vec4 color = texture(textures[0], uv);
    outColor = vec4(color.rgb, 1.0);
}
