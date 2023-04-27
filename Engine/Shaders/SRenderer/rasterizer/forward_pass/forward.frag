#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : require

#include "../../include/debug_draw.h"
#include "../../include/common_descriptor_sets.h"
#include "../../include/plugins/material/principled_common.h"
#include "../../../Utility/geometry.h"

layout(location = 0) in vec2 uv;
layout(location = 1) in flat uint matID;
layout(location = 2) in vec3 in_normalWS;
layout(location = 3) in vec4 tangentWS;
layout(location = 4) in vec4 posWS;

layout(location = 0) out vec4 outColor;

layout(binding = 0, set = 1) uniform sampler2DArray csm_depth;

struct PushConstants {
    uint lightIndex;
    float bias;
};
layout(push_constant) uniform PushConsts { 
    layout(offset = 4) PushConstants pushConstants; 
};

// float shadowCalculation(vec4 fragPosLightSpace, float ndl) {
//     vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
//     projCoords = projCoords * 0.5 + 0.5;
//     float closestDepth = texture(depth, projCoords.xy).r;
//     float currentDepth = (projCoords.z - 0.5) * 2.0;
//     float bias = max(0.05 * (1.0 - ndl), pushConstants.bias);
//     float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
//     return shadow;
// }

struct CascadeShadowmapData {
    mat4 cascade_transform[4];
    vec4 cascade_depths;
};
struct ShadowmapData {
    mat4 lightSpaceMatrix;
};
layout(binding = 1, set = 1, scalar) buffer _ShadowmapBuffer { ShadowmapData shadowmap_buffer[]; };
layout(binding = 2, set = 1, scalar) uniform _CascadeShadowmapBuffer { CascadeShadowmapData cascade_sm_info; };

vec3 cascade_debug() {
    vec4 fragPosViewSpace = globalUniform.cameraData.viewMat * posWS;
    float depthValue = abs(fragPosViewSpace.z);
    int layer = -1;
    for (int i = 0; i < 4; ++i) {
        if (depthValue < cascade_sm_info.cascade_depths[i]) {
            layer = i;
            break;
        }
    }
    if (layer == -1) {
        return vec3(1,0,1);
    }
    mat4 lightSpaceMatrix = cascade_sm_info.cascade_transform[layer];
    vec4 fragPosLightSpace = lightSpaceMatrix * posWS;
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    if(projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0) {
        return vec3(1,0,1);
    }
    if(projCoords.z < 0.0 || projCoords.z > 1.0) {
        return vec3(1,0,1);
    }

    if (layer == 0) {
        return vec3(0.9,0,0);
    }
    else if (layer == 1) {
        return vec3(0.0,0,0.9);
    }
    else if (layer == 2) {
        return vec3(0.5, 0.5 ,0);
    }
    else if (layer == 3) {
        return vec3(0.0,1,0.3);
    }
    return vec3(1,0,1);
}

float cascadeShadowCalculation(float ndl) {
    vec4 fragPosViewSpace = globalUniform.cameraData.viewMat * posWS;
    float depthValue = abs(fragPosViewSpace.z);
    int layer = -1;
    for (int i = 0; i < 4; ++i) {
        if (depthValue < cascade_sm_info.cascade_depths[i]) {
            layer = i;
            break;
        }
    }
    if (layer == -1) {
        layer = 3;
    }

    mat4 lightSpaceMatrix = cascade_sm_info.cascade_transform[layer];
    vec4 fragPosLightSpace = lightSpaceMatrix * posWS;
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    float currentDepth = (projCoords.z - 0.5) * 2.0;

    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if (currentDepth > 1.0) {
        return 1.0;
    }

    // calculate bias (based on depth map resolution and slope)
    float bias = max(0.05 * (1.0 - ndl), pushConstants.bias);
    const vec4 modifer_array = vec4(0.1, 0.3, 0.1, 0.1);
    const float biasModifier = modifer_array[layer];
    if(layer > 0)
        bias *= 1 / (cascade_sm_info.cascade_depths[layer] * biasModifier);
    else {
        bias *= 0.25;
    }
    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / vec2(textureSize(csm_depth, 0));
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y)  {
            float pcfDepth = texture(csm_depth, vec3(projCoords.xy + vec2(x, y) * texelSize, layer)).r;
            shadow += (currentDepth - bias) > pcfDepth ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;

    // if(projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0) {
    //     return 0;
    // }
    // if(projCoords.z < 0.0 || projCoords.z > 1.0) {
    //     return 0;
    // }
    // float closestDepth = texture(csm_depth, vec3(projCoords.xy, layer)).r;
    // float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
    // return shadow;

    return shadow;
}

void main() {
    mat3 TBN = buildTangentToWorld(tangentWS, in_normalWS);
    if(!gl_FrontFacing) TBN[2] = -TBN[2];
    vec3 normalWS = TBN[2];

    vec3 lightDir = normalize(-analytic_lights[pushConstants.lightIndex].direction);
    float shadow = cascadeShadowCalculation(dot(normalWS, lightDir));
    
    float diff = max(dot(normalWS, lightDir), 0.0);

    PrincipledMaterialData material = principled_materials[matID];
    vec3 base_color = texture(textures[material.basecolor_opacity_tex], uv).rgb;
    // float depth = texture(depth, uv).r;
    // vec3 normal = texture(textures[material.normal_bump_tex], uv).rgb;
    // normal = normalize(normal * 2.0 - 1.0);   
    // normal = normalize(TBN * normal);

    // // outColor = vec4(base_color, 1.0);
    outColor = vec4((1-shadow) * diff * base_color, 1.0);
    // outColor = vec4(cascade_debug(), 1.0);
}