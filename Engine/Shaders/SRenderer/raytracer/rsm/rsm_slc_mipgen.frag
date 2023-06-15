
#version 460
#extension GL_GOOGLE_include_directive : enable
#include "../../../Utility/geometry.h"

layout(location = 0) in vec2 in_uv;

layout(location = 0) out float sum_luminance;
layout(location = 1) out vec4  normalcone;
layout(location = 2) out vec4  boundingboxXY;   // min,max x,y 
layout(location = 3) out vec2  boundingboxZ;  	// min,max z

struct PushConstants { 
    ivec2 src_dim;
    ivec2 dst_dim;
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

layout(binding = 0) uniform sampler2D in_luminance;
layout(binding = 1) uniform sampler2D in_normalcone;
layout(binding = 2) uniform sampler2D in_boundingboxXY;
layout(binding = 3) uniform sampler2D in_boundingboxZ;

vec4 MergeCones(vec4 cone1, vec4 cone2) {
	vec4 ret;

    const bool swap = cone1.w >= cone2.w;
    vec3 axis_a = swap ? cone1.xyz : cone2.xyz;
    vec3 axis_b = swap ? cone2.xyz : cone1.xyz;
    float angle_a = swap ? cone1.w : cone2.w;
    float angle_b = swap ? cone2.w : cone1.w;

	float cosGamma = max(-1.f, min(1.f, dot(axis_a, axis_b)));
	float gamma = acos(cosGamma);

	if (cosGamma > 0.9999) {
		ret.xyz = axis_a;
		ret.w = min(angle_a, k_pi);
		return ret;
	}

	if (cosGamma < -0.9999) {
		ret.xyz = axis_a;
		ret.w = k_pi;
		return ret;
	}

	if (min(gamma + angle_b, k_pi) <= angle_a) {
		ret.xyz = axis_a;
		ret.w = angle_a;
		return ret;
	}

	ret.w = (angle_a + angle_b + gamma) / 2;

	if (ret.w >= k_pi) {
		ret.xyz = axis_a;
		ret.w = k_pi;
		return ret;
	}
    
	float rot = ret.w - angle_a;

	// slerp(axis_a, axis_b, rot / gamma);
	float t = rot / gamma;
	ret.xyz = (sin((1 - t)*gamma)*axis_a + sin(t*gamma)*axis_b) / sin(gamma);
    ret.xyz = normalize(ret.xyz);
	return ret;
}

vec4 MergeConesSafe(vec4 cone1, vec4 cone2) {
    bool invalid1 = (cone1 == vec4(0,0,0,0));
    bool invalid2 = (cone2 == vec4(0,0,0,0));
    if(invalid1 && invalid2) return vec4(0,0,0,0);
    if(invalid1) return cone2;
    if(invalid2) return cone1;
    return MergeCones(cone1, cone2);
}

void main() {
    const vec2 uv = in_uv;
    const vec2 ratio = vec2(pushConstants.src_dim) / pushConstants.dst_dim;

    const ivec2 vWriteCoord = ivec2(uv * pushConstants.dst_dim);
    const ivec2 vReadCoord = vWriteCoord << 1;

    const vec4 bounding_box_xy_0 = texelFetch(in_boundingboxXY, vReadCoord, 0);
    const vec4 bounding_box_xy_1 = texelFetch(in_boundingboxXY, vReadCoord + ivec2(1, 0), 0);
    const vec4 bounding_box_xy_2 = texelFetch(in_boundingboxXY, vReadCoord + ivec2(0, 1), 0);
    const vec4 bounding_box_xy_3 = texelFetch(in_boundingboxXY, vReadCoord + ivec2(1, 1), 0);

    const vec4 bounding_box_z_0 = texelFetch(in_boundingboxZ, vReadCoord, 0);
    const vec4 bounding_box_z_1 = texelFetch(in_boundingboxZ, vReadCoord + ivec2(1, 0), 0);
    const vec4 bounding_box_z_2 = texelFetch(in_boundingboxZ, vReadCoord + ivec2(0, 1), 0);
    const vec4 bounding_box_z_3 = texelFetch(in_boundingboxZ, vReadCoord + ivec2(1, 1), 0);

    const vec4 nc_pack_0 = texelFetch(in_normalcone, vReadCoord, 0);
    const vec4 nc_pack_1 = texelFetch(in_normalcone, vReadCoord + ivec2(1, 0), 0);
    const vec4 nc_pack_2 = texelFetch(in_normalcone, vReadCoord + ivec2(0, 1), 0);
    const vec4 nc_pack_3 = texelFetch(in_normalcone, vReadCoord + ivec2(1, 1), 0);

    const AABB aabb_0 = AABB(vec3(bounding_box_xy_0.xy, bounding_box_z_0.x), vec3(bounding_box_xy_0.zw, bounding_box_z_0.y));
    const AABB aabb_1 = AABB(vec3(bounding_box_xy_1.xy, bounding_box_z_1.x), vec3(bounding_box_xy_1.zw, bounding_box_z_1.y));
    const AABB aabb_2 = AABB(vec3(bounding_box_xy_2.xy, bounding_box_z_2.x), vec3(bounding_box_xy_2.zw, bounding_box_z_2.y));
    const AABB aabb_3 = AABB(vec3(bounding_box_xy_3.xy, bounding_box_z_3.x), vec3(bounding_box_xy_3.zw, bounding_box_z_3.y));
    
    const bool invalid_0 = bounding_box_xy_0.x == k_inf;
    const bool invalid_1 = bounding_box_xy_1.x == k_inf;
    const bool invalid_2 = bounding_box_xy_2.x == k_inf;
    const bool invalid_3 = bounding_box_xy_3.x == k_inf;

    const AABB aabb_01 = aabbUnion(aabb_0, aabb_1);
    const AABB aabb_23 = aabbUnion(aabb_2, aabb_3);
    const AABB aabb_03 = aabbUnion(aabb_01, aabb_23);

	// TODO: replace with gather here
    const vec4 lumin_samples = vec4(
        texelFetch(in_luminance, vReadCoord, 0).x,
        texelFetch(in_luminance, vReadCoord + ivec2(1, 0), 0).x,
        texelFetch(in_luminance, vReadCoord + ivec2(0, 1), 0).x,
        texelFetch(in_luminance, vReadCoord + ivec2(1, 1), 0).x
    );
    
    const vec4 normal_cone_01 = MergeConesSafe(nc_pack_0, nc_pack_1);
    const vec4 normal_cone_23 = MergeConesSafe(nc_pack_2, nc_pack_3);

    const float sum_lumin = dot(lumin_samples, vec4(1));

    const vec4 normal_cone_03 = MergeConesSafe(normal_cone_01, normal_cone_23);
    
	// Output
    normalcone = normal_cone_03;
    sum_luminance = sum_lumin;
    boundingboxXY = vec4(aabb_03.min.xy, aabb_03.max.xy);
    boundingboxZ = vec2(aabb_03.min.z, aabb_03.max.z);
}