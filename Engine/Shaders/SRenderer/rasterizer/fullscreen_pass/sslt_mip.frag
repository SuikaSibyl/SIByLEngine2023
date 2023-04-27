
#version 460
#extension GL_GOOGLE_include_directive : enable

layout(location = 0) in vec2 in_uv;

layout(location = 0) out float sum_luminance;
layout(location = 1) out vec4  boundingbox; // min,max x,y
layout(location = 2) out vec4  bbnc_pack;   // min,max x,y 
layout(location = 3) out vec4  normal_cone;  // min,max x,y 

struct PushConstants { 
    ivec2 src_dim;
    ivec2 dst_dim;
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

layout(binding = 0) uniform sampler2D in_luminance;
layout(binding = 1) uniform sampler2D in_boundingbox;
layout(binding = 2) uniform sampler2D in_bbnc_pack;
layout(binding = 3) uniform sampler2D in_normal;

#include "../../../Utility/geometry.h"

void main() {
    const vec2 uv = in_uv;
    const vec2 ratio = vec2(pushConstants.src_dim) / pushConstants.dst_dim;

    const ivec2 vWriteCoord = ivec2(uv * pushConstants.dst_dim);
    const ivec2 vReadCoord = vWriteCoord << 1;

    const vec4 bounding_box_0 = texelFetch(in_boundingbox, vReadCoord, 0);
    const vec4 bounding_box_1 = texelFetch(in_boundingbox, vReadCoord + ivec2(1, 0), 0);
    const vec4 bounding_box_2 = texelFetch(in_boundingbox, vReadCoord + ivec2(0, 1), 0);
    const vec4 bounding_box_3 = texelFetch(in_boundingbox, vReadCoord + ivec2(1, 1), 0);

    const vec4 bbnc_pack_0 = texelFetch(in_bbnc_pack, vReadCoord, 0);
    const vec4 bbnc_pack_1 = texelFetch(in_bbnc_pack, vReadCoord + ivec2(1, 0), 0);
    const vec4 bbnc_pack_2 = texelFetch(in_bbnc_pack, vReadCoord + ivec2(0, 1), 0);
    const vec4 bbnc_pack_3 = texelFetch(in_bbnc_pack, vReadCoord + ivec2(1, 1), 0);

    const vec3 normal_0 = unpackNormal(texelFetch(in_normal, vReadCoord, 0).xyz);
    const vec3 normal_1 = unpackNormal(texelFetch(in_normal, vReadCoord + ivec2(1, 0), 0).xyz);
    const vec3 normal_2 = unpackNormal(texelFetch(in_normal, vReadCoord + ivec2(0, 1), 0).xyz);
    const vec3 normal_3 = unpackNormal(texelFetch(in_normal, vReadCoord + ivec2(1, 1), 0).xyz);

    const AABB aabb_0 = AABB(vec3(bounding_box_0.xy, bbnc_pack_0.x), vec3(bounding_box_0.zw, bbnc_pack_0.y));
    const AABB aabb_1 = AABB(vec3(bounding_box_1.xy, bbnc_pack_1.x), vec3(bounding_box_1.zw, bbnc_pack_1.y));
    const AABB aabb_2 = AABB(vec3(bounding_box_2.xy, bbnc_pack_2.x), vec3(bounding_box_2.zw, bbnc_pack_2.y));
    const AABB aabb_3 = AABB(vec3(bounding_box_3.xy, bbnc_pack_3.x), vec3(bounding_box_3.zw, bbnc_pack_3.y));
    
    const bool invalid_0 = bounding_box_0.x == k_inf;
    const bool invalid_1 = bounding_box_1.x == k_inf;
    const bool invalid_2 = bounding_box_2.x == k_inf;
    const bool invalid_3 = bounding_box_3.x == k_inf;

    const AABB aabb_01 = aabbUnion(aabb_0, aabb_1);
    const AABB aabb_23 = aabbUnion(aabb_2, aabb_3);
    const AABB aabb_03 = aabbUnion(aabb_01, aabb_23);

    const vec4 lumin_samples = vec4(
        texelFetch(in_luminance, vReadCoord, 0).x,
        texelFetch(in_luminance, vReadCoord + ivec2(1, 0), 0).x,
        texelFetch(in_luminance, vReadCoord + ivec2(0, 1), 0).x,
        texelFetch(in_luminance, vReadCoord + ivec2(1, 1), 0).x
    );
    
    const NormalConeEx normal_cone_0 = NormalConeEx(normal_0, bbnc_pack_0.z, bbnc_pack_0.w, !invalid_0);
    const NormalConeEx normal_cone_1 = NormalConeEx(normal_1, bbnc_pack_1.z, bbnc_pack_1.w, !invalid_1);
    const NormalConeEx normal_cone_2 = NormalConeEx(normal_2, bbnc_pack_2.z, bbnc_pack_2.w, !invalid_2);
    const NormalConeEx normal_cone_3 = NormalConeEx(normal_3, bbnc_pack_3.z, bbnc_pack_3.w, !invalid_3);

    const NormalConeEx normal_cone_01 = coneUnion(normal_cone_0, normal_cone_1);
    const NormalConeEx normal_cone_23 = coneUnion(normal_cone_2, normal_cone_3);

    const float sum_lumin = dot(lumin_samples, vec4(1));

    const NormalConeEx normal_cone_03 = coneUnion(normal_cone_01, normal_cone_23);
    
    boundingbox = vec4(aabb_03.min.xy, aabb_03.max.xy);
    bbnc_pack = vec4(aabb_03.min.z, aabb_03.max.z, normal_cone_03.theta_e, normal_cone_03.theta_o);
    vec4 test = vec4(packNormal(normal_cone_03.direction), 1);
    sum_luminance = sum_lumin;
    normal_cone = test;

    if(normal_cone_03.valid == false) {
        normal_cone = vec4(vec3(0,0,0), 1);
    }
}