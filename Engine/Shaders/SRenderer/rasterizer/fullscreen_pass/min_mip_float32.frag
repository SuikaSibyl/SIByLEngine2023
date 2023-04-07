
#version 460
layout(location = 0) in vec2 in_uv;
layout(location = 0) out float out_color;

struct PushConstants { 
    ivec2 src_dim;
    ivec2 dst_dim;
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

layout(binding = 0) uniform sampler2D tex;

float texelFetchSrc(in const ivec2 coord) {
    ivec2 clamped_coord = ivec2(
        clamp(coord.x, 0, pushConstants.src_dim.x - 1),
        clamp(coord.y, 0, pushConstants.src_dim.y - 1)
    );
    return texelFetch(tex, clamped_coord, 0).x;
}

void main() {
    const vec2 uv = in_uv;
    const vec2 ratio = vec2(pushConstants.src_dim) / pushConstants.dst_dim;

    const ivec2 vWriteCoord = ivec2(uv * pushConstants.dst_dim);
    const ivec2 vReadCoord = vWriteCoord << 1;

    const vec4 depth_samples = vec4(
        texelFetchSrc(vReadCoord).x,
        texelFetchSrc(vReadCoord + ivec2(1, 0)).x,
        texelFetchSrc(vReadCoord + ivec2(0, 1)).x,
        texelFetchSrc(vReadCoord + ivec2(1, 1)).x
    );

    float min_depth = min(depth_samples.x, min(depth_samples.y, min(depth_samples.z, depth_samples.w)));

    const bool needExtraSampleX = ratio.x > 2;
    const bool needExtraSampleY = ratio.y > 2;

    min_depth = needExtraSampleX ? min(min_depth, min(
        texelFetchSrc(vReadCoord + ivec2(2, 0)).x, 
        texelFetchSrc(vReadCoord + ivec2(2, 1)).x
    )) : min_depth;

    min_depth = needExtraSampleY ? min(min_depth, min(
        texelFetchSrc(vReadCoord + ivec2(0, 2)).x, 
        texelFetchSrc(vReadCoord + ivec2(1, 2)).x
    )) : min_depth;

    min_depth = (needExtraSampleX && needExtraSampleY) 
        ? min(min_depth, texelFetchSrc(vReadCoord + ivec2(2, 2)).x) 
        : min_depth;

    out_color = min_depth;
}