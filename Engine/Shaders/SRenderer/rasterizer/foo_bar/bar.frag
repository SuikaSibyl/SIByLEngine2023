
#version 460
layout(location = 0) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

struct PushConstants {
    vec2    view_size;
    int     hiz_mip_levels;
    uint    max_iteration;
    int     strategy;
    int     sample_batch;
    int     hilum_mip_levels;
    float   max_thickness;
    mat4    InvProjMat;
    mat4    ProjMat;
    mat4    TransInvViewMat;
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

layout(binding = 0, set = 0) uniform sampler2D base_color;
layout(binding = 1, set = 0) uniform sampler2D depth_luminance;
layout(binding = 2, set = 0) uniform sampler2D ws_normal;

layout(binding = 0, set = 1) uniform sampler2D hi_luminance;
layout(binding = 1, set = 1) uniform sampler2D hi_z;
layout(binding = 2, set = 1) uniform sampler2D light_projection;

#extension GL_GOOGLE_include_directive : enable

#include "../fullscreen_pass/ssgi/ssgi.glsl"
#include "../../../Utility/random.h"

vec3 unpackNormal(in const vec3 normal) {
    return normalize(normal * 2.0 - 1.0);
}

ivec2 sampleHiLumin(inout uint RNG, out float pdf) {
    int mip_level = pushConstants.hilum_mip_levels - 1;
    ivec2 xy = ivec2(0, 0);
    float parent_importance = texelFetch(hi_luminance, xy, mip_level).x;
    
    ivec2 four_neighbors[4];
    four_neighbors[0] = ivec2(0, 0);
    four_neighbors[1] = ivec2(0, 1);
    four_neighbors[2] = ivec2(1, 0);
    four_neighbors[3] = ivec2(1, 1);

    float p = 1.f;
    
    for (int i = pushConstants.hilum_mip_levels-1; i>=0; --i) {
        if(i!=pushConstants.hilum_mip_levels-1) {
            xy *= 2;
        }
        mip_level--; // next mip level
        const float rnd = UniformFloat(RNG);  // sample next level
        float accum = 0.f;
        int last_non_zero = -1;
        float last_non_zero_imp = 0;
        float last_non_zero_pdf = 0;
        for (int j=0; j<4; ++j) {
            ivec2 xy_offset = four_neighbors[j];
            float importance = texelFetch(hi_luminance, xy + xy_offset, mip_level).x;
            float nimportance = importance / parent_importance;
            accum += nimportance;
            
            if(nimportance > 0) {
                last_non_zero = j;
                last_non_zero_pdf = nimportance;
                last_non_zero_imp = importance;
            }

            if(rnd < accum) {
                xy = xy + xy_offset;
                p = p * nimportance;
                parent_importance = importance;
                break;
            }
            else if(j==3 && last_non_zero!=-1) {
                xy = xy + four_neighbors[last_non_zero];
                p = p * last_non_zero_pdf;
                parent_importance = last_non_zero_imp;
                break;
            }
            else {
                // should not happen...
            }
        }
    }
    pdf = p;
    return xy;
}

void main() {
    const vec2 uv = in_uv;
    const uvec2 tid = uvec2(gl_FragCoord.xy);

    uint RNG = InitRNG(tid, pushConstants.sample_batch);
    // uint RNG = uint(tid.y * pushConstants.view_size.x + tid.x) + pushConstants.sample_batch;
    float light_sample_pdf;
    ivec2 light_sample = sampleHiLumin(RNG, light_sample_pdf);
    vec2 light_sample_uv = (vec2(light_sample) + 0.5) / 512;
    float light_depth = texture(depth_luminance, light_sample_uv).x;

    float sampleDepth = texelFetch(hi_z, ivec2(tid), 0).x;
    if(sampleDepth == 1) {
        out_color = vec4(vec3(0), 1);
        return;
    }
    vec4 samplePosInCS =  vec4(((vec2(tid)+0.5)/pushConstants.view_size)*2-1.0f, sampleDepth, 1);
    // samplePosInCS.y *= -1;

    vec3 vSampleNormalInWS = unpackNormal(texelFetch(ws_normal, ivec2(tid), 0).xyz);
    vec3 vSampleNormalInVS = normalize((pushConstants.TransInvViewMat * vec4(vSampleNormalInWS, 0)).xyz);
    vSampleNormalInVS.y = -vSampleNormalInVS.y;
    
    vec3  samplePosInTS;
    vec3  reflDirInTS;
    float maxDistance;
    ComputePosAndReflection(tid, vSampleNormalInVS, samplePosInTS, reflDirInTS, maxDistance);

    vec3 intersection;
    vec3 reflectedColor = vec3(0);
    if(FindIntersection_HiZ(samplePosInTS, reflDirInTS, maxDistance, intersection)) {
        reflectedColor = texture(base_color, intersection.xy).xyz;
    }

    vec4 samplePosInVS = pushConstants.InvProjMat * samplePosInCS;
    samplePosInVS /= samplePosInVS.w;
    const vec2  tst = pushConstants.view_size;
    // out_color = texture(base_color,\ uv);
    out_color = vec4(reflectedColor, 1);
}