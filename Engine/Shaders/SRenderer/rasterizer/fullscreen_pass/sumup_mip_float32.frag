#version 460
layout(location = 0) in vec2 in_uv;
layout(location = 0) out float out_color;

struct PushConstants { 
    uint src_size; 
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

layout(binding = 0) uniform sampler2D texSampler;

void main() {
    vec2 uv = vec2(in_uv.x, in_uv.y);
    const float pixel_width = 1. / pushConstants.src_size;
    const float half_pixel = 0.5 * pixel_width;
    vec2 lu = uv + vec2(-half_pixel, +half_pixel);
    float l0 = texture(texSampler, lu).r;
    vec2 ru = uv + vec2(+half_pixel, +half_pixel);
    float l1 = texture(texSampler, ru).r;
    vec2 ld = uv + vec2(-half_pixel, -half_pixel);
    float l2 = texture(texSampler, ld).r;
    vec2 rd = uv + vec2(+half_pixel, -half_pixel);
    float l3 = texture(texSampler, rd).r;
    
    out_color = l0 + l1 + l2 + l3;
}