#version 460
layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;
#extension GL_EXT_scalar_block_layout : require

struct PushConstants { 
    ivec2 dst_dim;
    float checkerboard_size;
    float padding;
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

layout(binding = 0, set = 0, scalar) buffer  _ColorBuffer  { vec3 colors[]; };

void main() {
    const vec2 uv = in_uv;
    const vec2 xy = uv * pushConstants.dst_dim;
    // checkerboard
    const vec2 pos = floor(xy / pushConstants.checkerboard_size);
    const float PatternMask = mod(pos.x + mod(pos.y, 2.0), 2.0);
    const float checkerboard = PatternMask * 1.0;
    const vec3 checker_col_0 = vec3(0.05, 0.05, 0.05);
    const vec3 checker_col_1 = vec3(0.10, 0.10, 0.10);
    const vec3 checkerboard_col = mix(checker_col_0, checker_col_1, checkerboard);

    const ivec2 ixy = ivec2(floor(xy));
    vec3 grid_color = colors[ixy.y + ixy.x * pushConstants.dst_dim.y];

    out_color = vec4(mix(checkerboard_col, grid_color, 0.5), 1.0);
}