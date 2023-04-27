
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
    float   min_thickness;
    float   padd0;
    float   padd1;
    float   padd2;
    vec4    iDebugPos;
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

#include "../ssrt/ssrt.glsl"
#include "../../../Utility/random.h"

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

const int grid = 5;
const float scale = 1.1 * float(grid-1);

vec2 StoQ(vec2 s, vec2 r) {
    return (s+s-r) * scale / r.y; // + .5;
}

void main() {
    const vec2 uv = in_uv;
    const uvec2 tid = uvec2(gl_FragCoord.xy);
    const vec2 iResolution = pushConstants.view_size;

    // vec2 q = StoQ(vec2(tid), iResolution);

    // ivec2 qi = ivec2(floor(q)); // gridcell
    // vec3 c = mix(vec3(.9), vec3(1), float((qi.x^qi.y)&1)); // checks bg

    // vec2 m0 = StoQ(abs(iMouse.zw), R);
    // vec2 m1 = StoQ(iMouse.xy, R);
    
    // qi = ivec2(floor(q)); // gridcell
    // celltouchesline = false;
    // pixel = q; cd = 3e38;
    // float l = ScanDDA2(m0, m1);

    // float l = ScanDDA2(m0, m1);

    // out_color = vec4(c, 1);

    vec3 sampleNormalInVS;
    RayStartInfo startInfo;
    bool hasIsect = unpackVSInfo(tid, sampleNormalInVS, startInfo);
    if(!hasIsect) {
        out_color = vec4(vec3(0), 1);
        return;
    }

    vec3 vCamToSampleInVS = normalize(startInfo.samplePosInVS.xyz);
    vec3 vRefDirInVS = normalize(reflect(vCamToSampleInVS.xyz, sampleNormalInVS.xyz));

    SSRay ray = PrepareSSRT(startInfo, vRefDirInVS);

    // // if (ray.rayDirInTS.z < 0) {
    // //     out_color = vec4(0,1,1, 1);
    // //     return;
    // // }

    vec3 intersection;
    vec3 reflectedColor = vec3(0);
    // int iter; vec4 u2v;
    if(FindIntersection_Linear(ray, intersection)) {
        reflectedColor = texture(base_color, intersection.xy).xyz;
    }

    out_color = vec4(reflectedColor, 1);
}