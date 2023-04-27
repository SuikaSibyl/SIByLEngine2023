
#version 460
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : enable
#include "../../include/definitions/camera.h"

layout(location = 0) in vec2 in_uv;

layout(location = 0) out float sum_luminance;
layout(location = 1) out vec4  boundingbox; // min,max x,y
layout(location = 2) out vec4  bbnc_pack;   // min,max x,y 
layout(location = 3) out vec4  normal_cone;  // min,max x,y 

struct PushConstants { 
    mat4  transInvViewMat;
    mat4  invProjMat;
    uvec2 resolution;
    int   importance_operator;
    bool  modulateJacobian;
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };
layout(binding = 3, set = 0, scalar) uniform _CameraUniforms { CameraData    gCamera; };

layout(binding = 0) uniform sampler2D in_color;
layout(binding = 1) uniform sampler2D in_depth;
layout(binding = 2) uniform sampler2D in_normal;

#include "../../include/spectrum.h"
#include "../../../Utility/geometry.h"

float computeJacobian_1(
    in const vec2 sample_uv,
    in const vec3 pos_vs,
    in const vec3 normal_vs
) {
    vec3 pix_position = gCamera.cameraW
                + (-0.5 + sample_uv.x) * 2 * gCamera.cameraU
                + (-0.5 + sample_uv.y) * 2 * gCamera.cameraV;
    vec3 pix_dir = normalize(pix_position);
    float cos_0 = max(dot(-pix_dir, -normalize(gCamera.cameraW)), 0);
    float r_0 = length(pix_position);
    float area_0 = 4 * length(gCamera.cameraU) * length(gCamera.cameraV);
    float jacobian_0 = cos_0 * area_0 / (r_0 * r_0);

    float dist_1 = length(pos_vs.xyz);
    vec3 dir_1 = normalize(pos_vs.xyz);
    float cos_1 = max(dot(normal_vs, -dir_1), 0);
    return jacobian_0 * (dist_1 * dist_1) / cos_1;
}

void main() {
    const vec2 uv = in_uv;
    const ivec2 id = ivec2(uv * pushConstants.resolution);

    // Unpack the depth from the texture.
    const float sampleDepth = texelFetch(in_depth, id, 0).x;
    const vec3 color = texture(in_color, uv).rgb;
    if(sampleDepth == 1) {
        boundingbox = vec4(vec2(k_inf), vec2(-k_inf));
        bbnc_pack = vec4(k_inf, -k_inf, 0, 0);
        normal_cone = vec4(0);
        sum_luminance = 0;
    }
    else {
        // Unpack the normal from the texture.
        const vec3 sampleNormalInWS = unpackNormal(texelFetch(in_normal, id, 0).xyz);
        vec3 sampleNormalInVS = normalize((pushConstants.transInvViewMat * vec4(sampleNormalInWS, 0)).xyz);
        sampleNormalInVS.y = -sampleNormalInVS.y;

        // From the depth, compute the position in clip space.
        vec4 samplePosInCS =  vec4(uv*2-1.0f, sampleDepth, 1);
        samplePosInCS.y *= -1;
        // From the depth, compute the position in view space.
        vec4 samplePosInVS = pushConstants.invProjMat * samplePosInCS;
        samplePosInVS /= samplePosInVS.w;

        float importance = 0;
        if(pushConstants.importance_operator == 0) // luminance
            importance = luminance(color);
        else if(pushConstants.importance_operator == 1) // average
            importance = dot(color, vec3(1)) / 3;
        else if(pushConstants.importance_operator == 2) // max
            importance = max(max(color.x, color.y), color.z);
        else if(pushConstants.importance_operator == 3) // uniform
            importance = 1;
            
        if(pushConstants.modulateJacobian) {
            const float jacobian = computeJacobian_1(uv, samplePosInVS.xyz, sampleNormalInVS);
            importance *= jacobian / 1000;
        }
        
        boundingbox = vec4(vec2(samplePosInVS.xy), vec2(samplePosInVS.xy));
        bbnc_pack = vec4(samplePosInVS.z, samplePosInVS.z, k_pi_over_2, 0);
        normal_cone = vec4(packNormal(sampleNormalInVS), 1);
        sum_luminance = importance;
    }
}