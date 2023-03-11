#version 460
layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 color_output;

struct PushConstants { 
    uvec2 resolution;
    uint sample_batch; 
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

layout(binding = 1, set = 2, r32f) coherent uniform image2DArray atomicRGB;

void main() {
    const ivec2 resolution = ivec2(pushConstants.resolution);
    const ivec2 pixel = ivec2((vec2(uv.x, 1-uv.y)) * vec2(resolution));
    float r = imageLoad(atomicRGB, ivec3(pixel, 0)).x;
    float g = imageLoad(atomicRGB, ivec3(pixel, 1)).x;
    float b = imageLoad(atomicRGB, ivec3(pixel, 2)).x;
    float a = imageLoad(atomicRGB, ivec3(pixel, 3)).x;
    color_output = vec4(vec3(r,g,b)/a, 1);
}