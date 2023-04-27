#version 460
layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

struct PushConstants {
    vec2    view_size;
    int     hiz_mip_levels;
    uint    max_iteration;
    int     strategy;
    int     sample_batch;
    uint    debug_ray_mode;
    float   max_thickness;
    uint    debug_mode;
    int	    mip_level;
    int     offset_steps;
    float   z_clamper;
    vec4    iDebugPos;
    float   z_min;
    float   z_range;
    float   padding0;
    float   padding1;
    mat4    InvProjMat;
    mat4    ProjMat;
    mat4    TransInvViewMat;
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

layout(binding = 0, set = 0) uniform sampler2D base_color;
layout(binding = 1, set = 0) uniform sampler2D depth_luminance;
layout(binding = 2, set = 0) uniform sampler2D ws_normal;
layout(binding = 3, set = 0) uniform sampler2D light_projection;
layout(binding = 4, set = 0) uniform sampler2D hi_luminance;
layout(binding = 5, set = 0) uniform sampler2D hi_z;

#extension GL_GOOGLE_include_directive : enable

#define DEBUG 1
#include "ssrt.glsl"
#include "../../../Utility/random.h"
#include "../../include/debug_draw.h"

const int grid = 8;
const float scale = 1.1 * float(grid-1);

vec2 StoQ(vec2 s, vec2 r) {
    return (s+s-r) * scale / r.y; // + .5;
}

// iq's line segment distance, trimmed & renamed
// just so I can see the mouse ray
float seg(vec2 p, vec2 a, vec2 b) {
	p -= a; b -= a;
    return length(p - b * clamp(dot(p, b) / dot(b, b), 0., 1.));
}


void overlay(inout vec3 o, vec3 c, float a) {
    o = mix(o, c, clamp(a, 0., 1.));
}

float LinearizeDepth(in const float depth)  {
    float zNear = 0.1;    // TODO: Replace by the zNear of your perspective projection
    float zFar  = 1000.0; // TODO: Replace by the zFar  of your perspective projection
    return (2.0 * zNear) / (zFar + zNear - depth * (zFar - zNear));
}

float visualizeZ(const float z) {
    float z_vis = LinearizeDepth(z);
    z_vis = (z_vis - pushConstants.z_min) / pushConstants.z_range;
    return z_vis;
}

vec3 specular() {
    const vec2 uv = in_uv;
    const uvec2 tid = uvec2(gl_FragCoord.xy);
    const vec2 iResolution = pushConstants.view_size;

    vec3 sampleNormalInVS;
    RayStartInfo startInfo;

    bool hasIsect = unpackVSInfo(tid, sampleNormalInVS, startInfo);
    if(!hasIsect) {
        return vec3(0);
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
    if(FindIntersection_HiZ(ray, intersection)) {
        reflectedColor = texture(base_color, intersection.xy).xyz;
    }
    return reflectedColor;
}

vec3 specularDDA() {
    const vec2 uv = in_uv;
    const uvec2 tid = uvec2(gl_FragCoord.xy);
    const vec2 iResolution = pushConstants.view_size;

    const int level = pushConstants.mip_level;
    const vec2 cellCount = getCellCount(level);

    vec3 sampleNormalInVS;
    RayStartInfo startInfo;

    bool hasIsect = unpackVSInfo(tid, sampleNormalInVS, startInfo);
    if(!hasIsect) {
        return vec3(0);
    }

    vec3 vCamToSampleInVS = normalize(startInfo.samplePosInVS.xyz);
    vec3 vRefDirInVS = normalize(reflect(vCamToSampleInVS.xyz, sampleNormalInVS.xyz));

    SSRay ray = PrepareSSRT(startInfo, vRefDirInVS);
    const vec3 samplePosInTS = ray.rayPosInTS;
    const vec3 vReflDirInTS = ray.rayDirInTS;
    const float maxTraceDistance = ray.maxDistance;

    hasIsect = false;
    vec3 intersection;

    const vec3 vReflectionEndPosInTS = samplePosInTS + vReflDirInTS * maxTraceDistance;
    const vec2 m0 = getCellCoord(samplePosInTS.xy, cellCount);
    const vec2 m1 = getCellCoord(vReflectionEndPosInTS.xy, cellCount);
    vec2 q = getCellCoord(tid/iResolution, cellCount);

    DebugPack debugPack;
    debugPack.qi = ivec2(floor(q)); // gridcell
    debugPack.celltouchesline = false;
    debugPack.pixel = q;
    debugPack.cd = 3e38;

    {
        DDA2D_Linear dda;
        Trace2D_Linear tr;
        setup(m0, m1, dda, tr);
        ivec2 n;
        vec3 pos_prev = samplePosInTS;
        vec2 uv_prev = tr.ro;
        const vec3 dir_norm = vReflDirInTS * abs(maxTraceDistance / tr.t1);

        float t_prev = 0.f;
        float z_prev = samplePosInTS.z;
        ivec2 prev_mp = dda.mp;
        float dir_weight = 1.f;
        {
            float ot = dda.t;
            int s;
            int iter = 0;
            while (true) {
                bool go = traverse(dda, tr, debugPack);
                
                vec3 pos = samplePosInTS + (dda.t + t_prev) * .5 * dir_norm;
                // vec2 uv = pos.xy * .5 + pos_prev.xy * .5;
                vec2 uv = tr.ro + tr.rd * (dda.t + t_prev) * .5;
                // float z = minimumDepthPlane(pos.xy, level, cellCount);
                float z = pos.z;
                float z_prevmp = minimumDepthPlane((prev_mp) / cellCount, level, cellCount);

                float z_a = (samplePosInTS + dda.t* dir_norm).z;
                float z_b = (samplePosInTS + t_prev* dir_norm).z;
                float alpha = (z_prevmp - z_a) / (z_b - z_a);
                vec3 pos_isect = samplePosInTS + ((dda.t * (1 - alpha)) + t_prev*alpha)* dir_norm;

                t_prev = dda.t;
                uv_prev = uv;

                prev_mp = dda.mp;
                if(!go) break;

                float thickness = z - z_prevmp;
                if(iter>pushConstants.offset_steps
                    && thickness > 0
                    && thickness < pushConstants.max_thickness
                ) {
                    hasIsect = true;
                    intersection = pos_isect;
                    break;
                }
                // if(iter>pushConstants.offset_steps
                //     && thickness >= 0
                //     && thickness < pushConstants.max_thickness) {
                //     hasIsect = true;
                //     intersection = pos;
                //     break;
                // }

                iter++;
                z_prev = z;
                s = nextIsect(dda, tr);
            }
            n = ivec2(0);
            n[s] = -tr.sd[s]; // avoid negating zeroes
        }
    }

    vec3 reflectedColor = vec3(0);
    if(hasIsect) {
        
        reflectedColor = texture(base_color, intersection.xy).xyz;
    }
    return reflectedColor;
}

vec3 debugSpecular() {
    const vec2 uv = in_uv;
    const uvec2 tid = uvec2(gl_FragCoord.xy);
    const vec2 iResolution = pushConstants.view_size;

    const int level = pushConstants.mip_level;
    const vec2 cellCount = getCellCount(level);

    vec3 sampleNormalInVS;
    RayStartInfo startInfo;
    bool hasIsect = unpackVSInfo(uvec2(pushConstants.iDebugPos.xy), sampleNormalInVS, startInfo);
    if(!hasIsect) {
        return vec3(0);
    }

    vec3 vCamToSampleInVS = normalize(startInfo.samplePosInVS.xyz);
    vec3 vRefDirInVS = normalize(reflect(vCamToSampleInVS.xyz, sampleNormalInVS.xyz));
    SSRay ray = PrepareSSRT(startInfo, vRefDirInVS);

    vec3 intersection;
    vec3 reflectedColor = vec3(0);

    float z_uv = minimumDepthPlane(uv, level, cellCount);
    vec3 c = vec3(visualizeZ(z_uv));

    // int iter; vec4 u2v;
    if(FindIntersection_HiZ(ray, intersection)) {
        vec2 start_uv = pushConstants.iDebugPos.xy / iResolution;
        float grayscale = draw_segment(uv, start_uv, intersection.xy, 0.01);
        overlay(c, vec3(0,1,0), grayscale);
    }
    else {
        vec3 endPoint = ray.rayPosInTS + ray.rayDirInTS * ray.maxDistance;
        vec2 start_uv = pushConstants.iDebugPos.xy / iResolution;
        float grayscale = draw_segment(uv, start_uv, endPoint.xy, 0.01);
        overlay(c, vec3(1,0,1), grayscale);
    }
    
    return c;
}

vec3 debugOcclusion() {
    const vec2 uv = in_uv;
    const uvec2 tid = uvec2(gl_FragCoord.xy);
    const vec2 iResolution = pushConstants.view_size;

    const int level = pushConstants.mip_level;
    const vec2 cellCount = getCellCount(level);

    vec3 sampleNormalInVS;
    RayStartInfo startInfo;
    bool hasIsect = unpackVSInfo(uvec2(pushConstants.iDebugPos.xy), sampleNormalInVS, startInfo);
    if(!hasIsect) {
        return vec3(0);
    }

    vec3 sampleNormalInVS_End;
    RayStartInfo endInfo;
    hasIsect = unpackVSInfo(uvec2(pushConstants.iDebugPos.zw), sampleNormalInVS_End, endInfo);
    if(!hasIsect) {
        return vec3(0);
    }
    SSRay ray = PrepareSSRT(startInfo, endInfo);

    vec3 reflectionDirVS = normalize((endInfo.samplePosInVS - startInfo.samplePosInVS).xyz);
    float cos_1 = dot(sampleNormalInVS, reflectionDirVS);
    float cos_2 = dot(sampleNormalInVS_End, -reflectionDirVS);
    ray.maxDistance = max(0, ray.maxDistance - 0.01);
    ray.minDistance = 0.01;

    vec3 intersection;
    vec3 reflectedColor = vec3(0);

    float z_uv = minimumDepthPlane(uv, level, cellCount);
    vec3 c = vec3(visualizeZ(z_uv));
    
    if(cos_1 < 0.01 || cos_2 < 0.01) {
        vec3 endPoint = ray.rayPosInTS + ray.rayDirInTS * ray.maxDistance;
        vec2 start_uv = pushConstants.iDebugPos.xy / iResolution;
        float grayscale = draw_segment(uv, start_uv, endPoint.xy, 0.002);
        overlay(c, vec3(1,0,1), grayscale);
    }
    else {
        // int iter; vec4 u2v;
        if(FindIntersection_HiZ(ray, intersection)) {
            vec2 start_uv = pushConstants.iDebugPos.xy / iResolution;
            float grayscale = draw_segment(uv, start_uv, intersection.xy, 0.002);
            overlay(c, vec3(1,0,0), grayscale);
        }
        else {
            vec3 endPoint = ray.rayPosInTS + ray.rayDirInTS * ray.maxDistance;
            vec2 start_uv = pushConstants.iDebugPos.xy / iResolution;
            float grayscale = draw_segment(uv, start_uv, endPoint.xy, 0.002);
            overlay(c, vec3(0,1,0), grayscale);
        }
    }
    
    return c;
}

vec3 debugMode() {
    const vec2 uv = in_uv;
    const uvec2 tid = uvec2(gl_FragCoord.xy);

    const vec2 iResolution = pushConstants.view_size;

    const int level = pushConstants.mip_level;
    const vec2 cellCount = getCellCount(level);
    // const ivec2 cell = ivec2(getCellCoord(uv, cellCount));


    // Mirror test 
    // --------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------
    vec3 sampleNormalInVS;
    RayStartInfo startInfo;
    bool hasIsect = unpackVSInfo(uvec2(pushConstants.iDebugPos.xy), sampleNormalInVS, startInfo);
    if(!hasIsect) {
        return vec3(0);
    }

    vec3 vRefDirInVS;
    SSRay ray;
    if(pushConstants.debug_mode == 2) {
        vec3 vCamToSampleInVS = normalize(startInfo.samplePosInVS.xyz);
        vRefDirInVS = normalize(reflect(vCamToSampleInVS.xyz, sampleNormalInVS.xyz));
        ray = PrepareSSRT(startInfo, vRefDirInVS);
    }
    else if(pushConstants.debug_mode == 3) {
        vec3 sampleNormalInVS_End;
        RayStartInfo endInfo;
        bool hasIsect = unpackVSInfo(uvec2(pushConstants.iDebugPos.zw), sampleNormalInVS_End, endInfo);
        if(!hasIsect) {
            return vec3(0);
        }
        ray = PrepareSSRT(startInfo, endInfo);
    }

    const vec3 samplePosInTS = ray.rayPosInTS;
    const vec3 vReflDirInTS = ray.rayDirInTS;
    const float maxTraceDistance = ray.maxDistance;

    const vec3 vReflectionEndPosInTS = samplePosInTS + vReflDirInTS * maxTraceDistance;

    const vec2 m0 = getCellCoord(samplePosInTS.xy, cellCount);
    const vec2 m1 = getCellCoord(vReflectionEndPosInTS.xy, cellCount);

    vec2 q = getCellCoord(tid/iResolution, cellCount);
    // vec2 m0 = getCellCoord(abs(pushConstants.iDebugPos.zw)/iResolution, cellCount);
    // vec2 m1 = getCellCoord(pushConstants.iDebugPos.xy/iResolution, cellCount);

    DebugPack debugPack;
    debugPack.qi = ivec2(floor(q)); // gridcell
    debugPack.celltouchesline = false;
    debugPack.pixel = q;
    debugPack.cd = 3e38;

    float ray_z_debug = 0;
    float ray_z_d_debug = 0;
    // float l = ScanDDA2(m0, m1, debugPack);
    float l = 0.f;
    DDA2D_Linear dda;
    Trace2D_Linear tr;
    setup(m0, m1, dda, tr);
    ivec2 n;
    vec3 pos_prev = samplePosInTS;
    vec2 uv_prev = tr.ro;
    const vec3 dir_norm = vReflDirInTS * abs(maxTraceDistance / tr.t1);

    bool hitSth = false;
    bool badRay = false;
    if(vReflDirInTS.z < 0 &&
        vReflDirInTS.z > -pushConstants.z_clamper ) {
            badRay = true;
    }

    float t_prev = 0.f;
    float z_prev = samplePosInTS.z;
    ivec2 prev_mp = dda.mp;
    float dir_weight = 1.f;
    {
        float ot = dda.t;
        int s;
        int iter = 0;
        while (true) {
            bool go = traverse(dda, tr, debugPack);
            
            vec3 pos = samplePosInTS + (dda.t + t_prev) * .5 * dir_norm;
            // vec2 uv = pos.xy * .5 + pos_prev.xy * .5;
            vec2 uv = tr.ro + tr.rd * (dda.t + t_prev) * .5;
            // float z = minimumDepthPlane(pos.xy, level, cellCount);
            float z = pos.z;
            float z_prevmp = minimumDepthPlane((prev_mp) / cellCount, level, cellCount);

            float z_a = (samplePosInTS + dda.t* dir_norm).z;
            float z_b = (samplePosInTS + t_prev* dir_norm).z;
            // if(z_a <= z_prevmp && z_b >= z_prevmp) {
            //     break;
            // }
            // float z = texelFetch(hi_z, dda.mp, 4).r;
            // pos_prev = pos;
            t_prev = dda.t;
            uv_prev = uv;

            if(debugPack.qi == prev_mp) {
                ray_z_debug = z_prevmp;
                ray_z_d_debug = z;
            }
            prev_mp = dda.mp;
            if(!go) break;

            // if(iter == pushConstants.offset_steps ) {
            //     if (z <= z_prevmp) {
            //         dir_weight = -1;
            //         break;
            //     }
            // }
            float thickness = z - z_prevmp;
            if(iter>pushConstants.offset_steps 
                && thickness >= 0
                && thickness < pushConstants.max_thickness
                && pushConstants.debug_ray_mode != 0) {
                hitSth = true;
                break;
            }

            iter++;
            z_prev = z;
            s = nextIsect(dda, tr);
        }
        n = ivec2(0);
        n[s] = -tr.sd[s]; // avoid negating zeroes
        l = vec2(ot, dda.t).y;
    }


    float z_uv = minimumDepthPlane(uv, level, cellCount);
    vec3 c = vec3(visualizeZ(z_uv));
    // vec3 c = mix(vec3(.9), vec3(1), float((debugPack.qi.x^debugPack.qi.y)&1)); // checks bg

    if (debugPack.celltouchesline) {
        // float z = minimumDepthPlane(vReflectionEndPosInTS.xy, 4, cellCount);
        // c = vec3(visualizeZ(ray_z_debug));
        // ray_z_d_debug
        // ray_z_debug
        c = vec3(visualizeZ(ray_z_d_debug));
        if(pushConstants.debug_ray_mode == 1) {
            if(badRay) {
                c = vec3(1,1,0);
            }
            else if (hitSth) {
                c.rb *= .8;
                c.g = max(c.g, .5);
            }
            else {
                c.gb *= .8;
                c.r = max(c.r, .5);
            }
        }
    }
    overlay(c, vec3(0,.4,0), 1. - .5*iResolution.y/scale * seg(q, m0, m1)); // segment

    float dotsize = .05; //dot(iMouse, vec4(1)) == 0. ? .05 : .03;
    float dotthick = .01;
    overlay(c, vec3(0,0,.0), .5 - .5*iResolution.y/scale*(abs(debugPack.cd - dotsize)-dotthick)); //(.5 - cd / scale * R.y)); // closest intersection point
    
    return c;
}

void main() {
    if(pushConstants.debug_mode == 0) {
        const vec2 uv = in_uv;
        // const ivec2 tid = ivec2(gl_FragCoord.xy);
        // out_color = vec4(texelFetch(base_color, tid, 0).rgb, 1);
        vec3 reflectedColor = texture(base_color, uv).xyz;
        out_color = vec4(reflectedColor, 1);

        // out_color = vec4(specularHiZ(), 1);
        // out_color = vec4(vec3(1), 1);
        // out_color = vec4(texture(base_color, vec2(0.5,0.5)).xyz, 1);
    }
    else if (pushConstants.debug_mode == 1) {
        out_color = vec4(specular(), 1);
        // const vec2 uv = in_uv;
        // out_color = vec4(texture(base_color, uv).xyz, 1);
        // out_color = vec4(vec3(0), 1);
    }
    else if (pushConstants.debug_mode == 2) {
        out_color = vec4(debugSpecular(), 1);
    }
    else if (pushConstants.debug_mode == 3) {
        out_color = vec4(debugOcclusion(), 1);
    }
    else {
        out_color = vec4(debugMode(), 1);
    }
}