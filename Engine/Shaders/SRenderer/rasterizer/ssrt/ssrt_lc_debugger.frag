#version 460
layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : enable

#include "../../../Utility/random.h"
#include "../../../Utility/geometry.h"
#include "../../../Utility/sampling.h"
#include "../../include/debug_draw.h"
#include "../../include/definitions/camera.h"

struct UniformStruct {
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
    int     is_depth;
    int     lightcut_mode;
    mat4    InvProjMat;
    mat4    ProjMat;
    mat4    TransInvViewMat;
};
layout(binding = 8, set = 0, scalar) uniform _Uniforms       { UniformStruct gUniform; };

struct GlobalUniforms {
  CameraData cameraData;
};
layout(binding = 9, set = 0, scalar) uniform _GlobalUniforms { GlobalUniforms globalUniform; };

layout(binding = 0, set = 0) uniform sampler2D base_color;
layout(binding = 1, set = 0) uniform sampler2D hi_z;
layout(binding = 2, set = 0) uniform sampler2D ws_normal;
layout(binding = 3, set = 0) uniform sampler2D importance_mip;
layout(binding = 4, set = 0) uniform sampler2D boundingbox_mip;
layout(binding = 5, set = 0) uniform sampler2D bbncpack_mip;
layout(binding = 6, set = 0) uniform sampler2D normalcone_mip;
layout(binding = 7, set = 0) uniform sampler2D di;

// #define DEBUG 1
#include "ssrt.glsl"

bool FindIntersection_Interface(
    in const SSRay ray,
    out vec3 intersection,
    inout SSRTConfig config
) {
    if(gUniform.debug_ray_mode == 0) {
        return FindIntersection_HiZ(ray, intersection, config);
    }
    else if(gUniform.debug_ray_mode == 1) {
        return FindIntersection_DDA(ray, intersection, config);
    }
    else if(gUniform.debug_ray_mode == 2) {
        return FindIntersection_DDA_Connected(ray, intersection, config);
    }
    else if(gUniform.debug_ray_mode == 3){
        return FindIntersection_DDA_Trianglulate_Safe(ray, intersection, config);
    }
}

#include "ssrt_sampler.glsl"

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
    z_vis = (z_vis - gUniform.z_min) / gUniform.z_range;
    return z_vis;
}

vec3 specular() {
    const vec2 uv = in_uv;
    const uvec2 tid = uvec2(gl_FragCoord.xy);
    const vec2 iResolution = gUniform.view_size;

    RayStartInfo startInfo;

    bool hasIsect = unpackVSInfo(tid, startInfo);
    if(!hasIsect) {
        return vec3(0);
    }

    vec3 vCamToSampleInVS = normalize(startInfo.samplePosInVS.xyz);
    vec3 vRefDirInVS = normalize(reflect(vCamToSampleInVS.xyz, startInfo.sampleNormalInVS.xyz));
    
    SSRay ray = PrepareSSRT(startInfo, vRefDirInVS);

    vec3 intersection;
    vec3 reflectedColor = vec3(0);
    SSRTConfig config = SSRTConfig(false);
    // int iter; vec4 u2v;
    vec3[4] points = FetchRectPointsSS(ivec2(tid / 2) + ivec2(1));
    ivec2 mod2 = ivec2(tid) % 2;
    
    const float z = texelFetch(hi_z, ivec2(tid), 0).r;  
    vec3 samplePositionInWS = SS2WS((tid + 0.5), z);    
    vec3 sampleNormalInWS = unpackNormal(texelFetch(ws_normal, ivec2(tid), 0).xyz);
    vec3 vCamToSampleInWS = normalize(samplePositionInWS - globalUniform.cameraData.posW);
    vec3 vRefDirInWS = normalize(reflect(vCamToSampleInWS, sampleNormalInWS));

    if(FindIntersection_Interface(ray, intersection, config
        // ,samplePositionInWS, vRefDirInWS
    )) {
        // reflectedColor = vec3(1,0,1);
        ivec2 xy = ivec2(intersection.xy * gUniform.view_size);
        reflectedColor = texelFetch(di, xy, 0).xyz;
    }
    return reflectedColor;
}

vec2 sampleImportanceUV(inout uint RNG, out float probability) {
    // gImportanceMIP
    int mip_level = 11;
    vec2 uv = vec2(0.5, 0.5);
    float parent_importance = textureLod(importance_mip, uv, mip_level).x;

    float p = 1.f;
    float pixel_size = 1. / 2;
    
    for (int i = 0; i < gUniform.is_depth; ++i) {
        pixel_size /= 2;    // half pixel size
        mip_level--; // next mip level
        // sample next level
        float rnd = UniformFloat(RNG);
        float accum = 0.f;
        int last_non_zero = -1;
        float last_non_zero_imp = 0;
        float last_non_zero_pdf = 0;
        for (int j=0; j<4; ++j) {
            vec2 uv_offset = four_neighbors[j] * pixel_size;
            float importance = textureLod(importance_mip, uv + uv_offset, mip_level).x;
            float nimportance = importance / parent_importance;
            accum += nimportance;
            
            if(nimportance > 0) {
                last_non_zero = j;
                last_non_zero_pdf = nimportance;
                last_non_zero_imp = importance;
            }

            if(rnd < accum) {
                uv = uv + uv_offset;
                p = p * nimportance;
                parent_importance = importance;
                break;
            }
            else if(j==3 && last_non_zero!=-1) {
                uv = uv + four_neighbors[last_non_zero] * pixel_size;
                p = p * last_non_zero_pdf;
                parent_importance = last_non_zero_imp;
                break;
            }
            else {
                // should not happen...
            }
        }
    }

    vec2 uv_pertub = vec2(UniformFloat(RNG), UniformFloat(RNG)); // (0,1)
    uv_pertub = vec2(-1, -1) + uv_pertub * 2; // (-1, 1)
    uv += uv_pertub * pixel_size;
    p /= (4 * pixel_size * pixel_size);
    probability = p;
    return uv;
}

float computeJacobian_1(in const vec2 sample_uv) {
    vec3 sampleNormalInVS_End;
    RayStartInfo endInfo;
    bool hasIsect = unpackVSInfo(uvec2(sample_uv * gUniform.view_size), endInfo);
    sampleNormalInVS_End = endInfo.sampleNormalInVS;
    if (!hasIsect) {
        return 0;
    }

    vec3 pix_position = globalUniform.cameraData.cameraW
                + (-0.5 + sample_uv.x) * 2 * globalUniform.cameraData.cameraU
                + (-0.5 + sample_uv.y) * 2 * globalUniform.cameraData.cameraV;
    vec3 pix_dir = normalize(pix_position);
    float cos_0 = max(dot(-pix_dir, -normalize(globalUniform.cameraData.cameraW)), 0);
    float r_0 = length(pix_position);
    float area_0 = 4 * length(globalUniform.cameraData.cameraU) * length(globalUniform.cameraData.cameraV);
    float jacobian_0 = cos_0 * area_0 / (r_0 * r_0);

    float dist_1 = length(endInfo.samplePosInVS.xyz);
    vec3 dir_1 = normalize(endInfo.samplePosInVS.xyz);
    float cos_1 = max(dot(sampleNormalInVS_End, -dir_1), 0);
    return jacobian_0 * (dist_1 * dist_1) / cos_1;
}

vec3 diffuse() {
    const vec2 uv = in_uv;
    const uvec2 tid = uvec2(gl_FragCoord.xy);
    const vec2 iResolution = gUniform.view_size;
    uint RNG = InitRNG(tid, gUniform.sample_batch);

    InitStateTS startState;
    bool hasIsect = unpackVSInfo(tid, startState);
    if(!hasIsect) {
        return vec3(0);
    }

    if(gUniform.strategy == 0) {
        SampleTS sample_cos = SampleTech_CosWeight(startState, RNG);
        return sample_cos.radiance;
    }
    else if(gUniform.strategy == 1) {
        SampleTS sample_uv = SampleTech_LightCut(startState, RNG);
        return sample_uv.radiance;
    }
    else if(gUniform.strategy == 2){
        vec3 radiance = vec3(0.f);

        SampleTS sample_cos = SampleTech_CosWeight(startState, RNG);
        if(sample_cos.hasIsect) {
            const float pcos_1 = sample_cos.pdf;
            const float pcos_2 = SampleTechPdf_LightCut(startState, sample_cos);
            if(isnan(pcos_2)) {
                radiance += sample_cos.radiance;
            }
            else {
                radiance += (pcos_1 / (pcos_1 + pcos_2)) * sample_cos.radiance;
            }
            // radiance += (pcos_1 / (pcos_1 + pcos_2)) * sample_cos.radiance;
            // radiance = vec3(pcos_2 * 100, 0, 0);
        }
        
        SampleTS sample_uv = SampleTech_LightCut(startState, RNG);
        if(sample_uv.hasIsect) {
            const float puv_1 = SampleTechPdf_CosWeight(startState, sample_uv);
            const float puv_2 = sample_uv.pdf;
            radiance += (puv_2 / (puv_1 + puv_2)) * sample_uv.radiance;
        }

        return radiance;
    }
}

vec3 debugSpecular() {
    const vec2 uv = in_uv;
    const uvec2 tid = uvec2(gl_FragCoord.xy);
    const vec2 iResolution = gUniform.view_size;

    const int level = gUniform.mip_level;
    const vec2 cellCount = getCellCount(level);

    vec3 sampleNormalInVS;
    RayStartInfo startInfo;
    bool hasIsect = unpackVSInfo(uvec2(gUniform.iDebugPos.xy), startInfo);
    sampleNormalInVS = startInfo.sampleNormalInVS;

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

    SSRTConfig config = SSRTConfig(false);
    // int iter; vec4 u2v;
    if(FindIntersection_Interface(ray, intersection, config)) {
        vec2 start_uv = gUniform.iDebugPos.xy / iResolution;
        float grayscale = draw_segment(uv, start_uv, intersection.xy, 0.002);
        overlay(c, vec3(0,1,0), grayscale);
    }
    else {
        vec3 endPoint = ray.rayPosInTS + ray.rayDirInTS * ray.maxDistance;
        vec2 start_uv = gUniform.iDebugPos.xy / iResolution;
        float grayscale = draw_segment(uv, start_uv, endPoint.xy, 0.002);
        overlay(c, vec3(1,0,1), grayscale);
    }
    
    return c;
}

vec3 debugOcclusion() {
    const vec2 uv = in_uv;
    const uvec2 tid = uvec2(gl_FragCoord.xy);
    const vec2 iResolution = gUniform.view_size;

    const int level = gUniform.mip_level;
    const vec2 cellCount = getCellCount(level);

    vec3 sampleNormalInVS;
    RayStartInfo startInfo;
    SSRTConfig config = SSRTConfig(false);
    config.normal_check= true;
    int debug_ray_status =  0;
    vec3 debug_value;
    bool hasIntersection = false;
    vec2 intersectionPos = vec2(0);

    float z_uv = minimumDepthPlane(uv, level, cellCount);
    vec3 c = vec3(visualizeZ(z_uv));

    bool hasIsect = unpackVSInfo(uvec2(gUniform.iDebugPos.xy), startInfo);
    sampleNormalInVS = startInfo.sampleNormalInVS;

    if(!hasIsect) {
        vec2 start_uv = gUniform.iDebugPos.xy / iResolution;
        vec2 end_uv = gUniform.iDebugPos.zw / iResolution;
        float first_seg = draw_segment(uv, start_uv, end_uv, 0.002);
        overlay(c, vec3(0,0,0), first_seg);
        return c;
    }
    else {
        vec3 sampleNormalInVS_End;
        RayStartInfo endInfo;
        hasIsect = unpackVSInfo(uvec2(gUniform.iDebugPos.zw), endInfo);
        sampleNormalInVS_End = endInfo.sampleNormalInVS;
        if(!hasIsect) {
            vec2 start_uv = gUniform.iDebugPos.xy / iResolution;
            vec2 end_uv = gUniform.iDebugPos.zw / iResolution;
            float first_seg = draw_segment(uv, start_uv, end_uv, 0.002);
            overlay(c, vec3(0), first_seg);
            return c;
        }

        const vec3 reflectionDirVS = normalize(endInfo.samplePosInVS.xyz - startInfo.samplePosInVS.xyz);
        InitStateTS initState = InitStateTS(startInfo, sampleNormalInVS);
        SampleTS isect;
        uint result = FindIntersection(initState, reflectionDirVS, isect);
        
        if(result == EnumIsectResult_None) {
            vec2 start_uv = gUniform.iDebugPos.xy / iResolution;
            vec2 end_uv = gUniform.iDebugPos.zw / iResolution;
            float first_seg = draw_segment(uv, start_uv, end_uv, 0.002);
            overlay(c, vec3(0.1), first_seg);
            return c;
        }
        else if(result == EnumIsectResult_True) {
            vec2 start_uv = gUniform.iDebugPos.xy / iResolution;
            vec2 isect_uv = isect.uv;
            vec2 end_uv = gUniform.iDebugPos.zw / iResolution;
            float first_seg = draw_segment(uv, start_uv, isect_uv, 0.002);
            float second_seg = draw_segment(uv, isect_uv, end_uv, 0.002);
            overlay(c, vec3(0,1,0), first_seg);
            overlay(c, vec3(0,0,1), second_seg);
            return c;
        }
        else if(result == EnumIsectResult_MismatchDir) {
            vec2 start_uv = gUniform.iDebugPos.xy / iResolution;
            vec2 isect_uv = isect.uv;
            vec2 end_uv = gUniform.iDebugPos.zw / iResolution;
            float first_seg = draw_segment(uv, start_uv, isect_uv, 0.002);
            float second_seg = draw_segment(uv, isect_uv, end_uv, 0.002);
            overlay(c, vec3(0,1,1), first_seg);
            overlay(c, vec3(1,1,0), second_seg);
            return c;
        }
        else if(result == EnumIsectResult_NegativeCos) {
            vec2 start_uv = gUniform.iDebugPos.xy / iResolution;
            vec2 end_uv = gUniform.iDebugPos.zw / iResolution;
            float first_seg = draw_segment(uv, start_uv, end_uv, 0.002);
            overlay(c, vec3(1,0,1), first_seg);
            return c;
        }
        else if(result == EnumIsectResult_Err) {
            vec2 start_uv = gUniform.iDebugPos.xy / iResolution;
            vec2 end_uv = gUniform.iDebugPos.zw / iResolution;
            float first_seg = draw_segment(uv, start_uv, end_uv, 0.002);
            overlay(c, vec3(0.5,0.,0.), first_seg);
            return c;

        }
    }
    
    return c;
}

vec3 showNormalCone() {
    const vec2 uv = in_uv;
    const uvec2 tid = uvec2(gl_FragCoord.xy);
    const vec2 iResolution = gUniform.view_size;

    const int level = gUniform.mip_level;
    const vec2 cellCount = getCellCount(level);
    const ivec2 cell = ivec2(getCellCoord(uv, cellCount));

    return unpackNormal(texelFetch(normalcone_mip, cell, level).xyz);
}

vec3 calcShowImportance() {
    const vec2 uv = in_uv;
    const uvec2 tid = uvec2(gUniform.iDebugPos.xy);

    InitStateTS startState;
    bool hasIsect = unpackVSInfo(tid, startState);
    if(!hasIsect) {
        return vec3(0);
    }

    SampleTS samplets;
    samplets.xy = ivec2(uv * gUniform.view_size);
    samplets.uv = uv;
    hasIsect = unpackVSInfo(samplets.xy, samplets.info);
    samplets.normalInVS = samplets.info.sampleNormalInVS;

    if(!hasIsect) {
        return vec3(0);
    }
    
    if(gUniform.strategy == 0) {
        float jacobian;
        texture_sample_jacobian(uv, startState, jacobian);
        const float pdf_gs = SampleTechPdf_CosWeight(startState, samplets);
        return vec3(pdf_gs * jacobian) / gUniform.iDebugPos.z;
    }
    else if(gUniform.strategy == 1) {
        float uv_pdf = sampleImportanceUV_pdf(
            samplets.uv,
            startState.info,
            startState.normalInVS
        );
        if(isnan(uv_pdf)) uv_pdf = 0.f;
        return vec3(uv_pdf) / gUniform.iDebugPos.z;
    }
    else if(gUniform.strategy == 2){
        float uv_pdf = sampleImportanceUV_pdf(
            samplets.uv,
            startState.info,
            startState.normalInVS
        );
        if(isnan(uv_pdf)) uv_pdf = 0.f;
        float jacobian;
        texture_sample_jacobian(uv, startState, jacobian);
        const float pdf_gs = SampleTechPdf_CosWeight(startState, samplets);
        return vec3(0.5 * (pdf_gs * jacobian + uv_pdf)) / gUniform.iDebugPos.z;
    }
    else if(gUniform.strategy == 3){
        float jacobian;
        vec3 radiance = texture_sample_jacobian(uv, startState, jacobian);
        const float luminance = dot(radiance, vec3(1));
        return vec3(luminance  / gUniform.iDebugPos.w);
    }
    return vec3(0);
}

vec3 showImportance() {
    vec3 c = calcShowImportance();
    if(isnan(c.x)) c = vec3(0);

    const vec2 uv = in_uv * gUniform.view_size;
    const vec2 tid = vec2(gUniform.iDebugPos.xy) + vec2(0.5);
    float point = distance(uv, tid) < 2 ? 1 : 0;
    overlay(c, vec3(0,1,0), point);

    return c;
}

void main() {
    // DI result
    if(gUniform.debug_mode == 0) {
        const vec2 uv = in_uv;
        vec3 reflectedColor = texture(di, uv).xyz;
        out_color = vec4(reflectedColor, 1);
        return;
    }
    else if (gUniform.debug_mode == 1) {
        const vec2 uv = in_uv;
        const vec3 di = texture(di, uv).xyz;
        const vec3 th = texture(base_color, uv).xyz;
        // out_color = vec4(th * specular() * 0.5 + di, 1);
        out_color = vec4(specular(), 1);
        return;
    }
    else if (gUniform.debug_mode == 2) {
        const vec2 uv = in_uv;
        // const vec3 th = texture(base_color, uv).xyz;
        out_color = vec4( diffuse(), 1);
        return;
    }
    else if (gUniform.debug_mode == 3) {
        out_color = vec4(debugSpecular(), 1);
        return;
    }
    else if (gUniform.debug_mode == 4) {
        out_color = vec4(debugOcclusion(), 1);
        return;
    }
    else if (gUniform.debug_mode == 5) {
        const uvec2 tid = uvec2(gl_FragCoord.xy);
        InitStateTS startState;
        bool hasIsect = unpackVSInfo(tid, startState);
        if(!hasIsect) {
            out_color = vec4(0);
        }else {
            out_color = vec4(startState.normalInVS, 1);
        }

        // out_color = vec4(showNormalCone(), 1);
        // const vec2 uv = in_uv;
        // float a = textureLod(importance_mip, uv, gUniform.mip_level).x;
        // out_color = vec4(a, 0, 0, 1);
        return;
    }
    else if (gUniform.debug_mode == 6) {
        const vec2 uv = in_uv;
        float j = computeJacobian_1(uv);
        out_color = vec4(j, 0, 0, 1);

        const vec2 iResolution = gUniform.view_size;
        vec3 sampleNormalInVS;
        RayStartInfo startInfo;
        bool hasIsect = unpackVSInfo(uvec2(uv*iResolution), startInfo);
        sampleNormalInVS = startInfo.sampleNormalInVS;
        if(!hasIsect) {
            out_color = vec4(vec3(0), 1);
        }
        else {
            out_color = vec4(packNormal(sampleNormalInVS), 1);
        }
        return;
    }
    else if (gUniform.debug_mode == 7) {
        out_color = vec4(showImportance(), 1);
    }
}