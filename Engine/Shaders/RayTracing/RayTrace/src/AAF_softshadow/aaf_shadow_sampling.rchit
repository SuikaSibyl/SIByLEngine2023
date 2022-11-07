#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "aaf_common.h"
#include "aaf_payloads.h"
#include "../../include/closestHitCommon.h"
#include "../../../../Utility/random.h"

layout(location = 0) rayPayloadInEXT InitalSamplePayload primaryPayLoad;
layout(location = 1) rayPayloadEXT ShadowRayPayload shadowPayLoad;
layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;

const float phongExp = 100;

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

    if(true) {
        const vec3 L = normalize(toLight);
        const float NdL = max(dot(hitInfo.worldNormal, L),0.0f);
        const vec3 H = normalize(L - normalize(gl_WorldRayDirectionEXT));
        const float NdH = max(dot(hitInfo.worldNormal, H),0.0f);
        vec3 color = vec3(0);
        // white light
        color += Kd * NdL;
        // do not ocnsider specular now
        // if (NdH > 0)
        //     color += Ks * pow(NdH, phongExp);
        primaryPayLoad.brdf = color;
    }

    const int sqrtShadowRaySamples = 3;
    const float invNSamples = 1.f/sqrtShadowRaySamples;
    vec2 lightSampleSeed;
    for(int i=0; i<sqrtShadowRaySamples; ++i) {
        for(int j=0; j<sqrtShadowRaySamples; ++j) {
            // stratified sampling the light
            lightSampleSeed.x = invNSamples* (i + stepAndOutputRNGFloat(primaryPayLoad.rngState));
            lightSampleSeed.y = invNSamples* (j + stepAndOutputRNGFloat(primaryPayLoad.rngState));
            const vec3 lightSample = sampleAreaLight(lightSampleSeed);
            const vec3 center2Sample = lightCenter - lightSample;
            const float strength = exp(-0.5 * dot(center2Sample,center2Sample) / (lightSigma * lightSigma));
            const vec3 lightDir = lightSample - secondaryRayOrigin;

            if(dot(hitInfo.worldNormal, lightDir) > 0.f) {
                // accumulate weights for vis
                primaryPayLoad.accumWeights += strength;
                // cast shadow ray
                shadowPayLoad.hit = false;
                shadowPayLoad.distanceMax = 0;
                shadowPayLoad.distanceMin = distanceToLight;
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
                // if hit any shadow caster
                if(shadowPayLoad.hit) {
                    primaryPayLoad.rayShadow = true;
                    const float d2min = min(max(distanceToLight - shadowPayLoad.distanceMax, 0), distanceToLight);
                    const float d2max = min(max(distanceToLight - shadowPayLoad.distanceMin, 0), distanceToLight);
                    const float s1 = distanceToLight/d2min - 1.0;
                    const float s2 = distanceToLight/d2max - 1.0;
                    primaryPayLoad.s1 = max(primaryPayLoad.s1, s1);
                    primaryPayLoad.s2 = min(primaryPayLoad.s2, s2);
                }
                // if shadow ray hit no objects
                else {
                    primaryPayLoad.accumVis += strength;
                }
            }
        }
    }
    // write to primary ray payload
    primaryPayLoad.rayHitSky = false;
    primaryPayLoad.hitPoint = hitInfo.worldPosition;
    primaryPayLoad.hitNormal = hitInfo.worldNormal;
}