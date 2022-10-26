#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "aaf_payloads.h"
#include "../../include/closestHitCommon.h"
#include "../../../../Utility/random.h"

layout(location = 0) rayPayloadInEXT InitalSamplePayload primaryPayLoad;
layout(location = 1) rayPayloadEXT ShadowRayPayload shadowPayLoad;
layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;

const vec3 lightMin = vec3(-0.24, 1.979, -0.22);
const vec3 lightMax = vec3(-0.24 + 0.47, 1.979, -0.22 + 0.38);
const vec3 lightCenter = 0.5 * (lightMin + lightMax);

vec3 sampleAreaLight(inout uint rngState) {
    vec2 lsample = vec2(stepAndOutputRNGFloat(rngState), stepAndOutputRNGFloat(rngState));
    vec2 xz = vec2(-0.24, -0.22) + lsample * vec2(0.47, 0.38);
    return vec3(xz.x, 1.979, xz.y);
}

vec3 sampleAreaLight(in vec2 lsample) {
    vec2 xz = vec2(-0.24, -0.22) + lsample * vec2(0.47, 0.38);
    return vec3(xz.x, 1.979, xz.y);
}

void main() {
    // get primary hit info
    HitInfo hitInfo = getObjectHitInfo();
    const vec3 secondaryRayOrigin = offsetPositionAlongNormal(hitInfo.worldPosition, hitInfo.worldNormal);

    const vec3 toLight = lightCenter - secondaryRayOrigin;
    const float distanceToLight = length(toLight);
    // compute BRDF in first pass ?

    shadowPayLoad.hit = false;
    shadowPayLoad.attenuation = vec3(1,0,0);
    shadowPayLoad.distanceMin = 9999;
    shadowPayLoad.distanceMax = 0;

    const float lightSigma = 1.f;
    const int sqrtShadowRaySamples = 3;
    const float invNSamples = 1.f/sqrtShadowRaySamples;
    vec2 lightSampleSeed;
    for(int i=0; i<sqrtShadowRaySamples; ++i) {
        for(int j=0; j<sqrtShadowRaySamples; ++j) {
            lightSampleSeed.x = invNSamples* (i + stepAndOutputRNGFloat(primaryPayLoad.rngState));
            lightSampleSeed.y = invNSamples* (j + stepAndOutputRNGFloat(primaryPayLoad.rngState));
            vec3 lightSample = sampleAreaLight(lightSampleSeed);
            vec3 center2Sample = lightCenter - lightSample;
            float strength = exp(-0.5 * dot(center2Sample,center2Sample) / (lightSigma * lightSigma));
            vec3 lightDir = lightCenter - secondaryRayOrigin;

            if(dot(hitInfo.worldNormal, lightDir) > 0.f) {
                // shadowPayLoad.attenuation = vec3(0,0,1);
                uint  flags = gl_RayFlagsNoneEXT;
                traceRayEXT(tlas,   // acceleration structure
                        flags,       // rayFlags
                        0xFF,        // cullMask
                        1,           // sbtRecordOffset
                        0,           // sbtRecordStride
                        1,           // missIndex
                        secondaryRayOrigin,  // ray origin
                        0.0,         // ray min range
                        normalize(lightDir),      // ray direction
                        length(lightDir),        // ray max range
                        1            // payload (location = 1)
                );
            }
        }
    }
    // write to primary ray payload
    primaryPayLoad.rayHitSky = false;
    primaryPayLoad.rayShadow = shadowPayLoad.hit;
    primaryPayLoad.distanceMin = shadowPayLoad.distanceMin;
    primaryPayLoad.distanceMax = shadowPayLoad.distanceMax;
}