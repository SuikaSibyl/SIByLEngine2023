#version 460
#extension GL_GOOGLE_include_directive : require

#include "../../../../Utility/random.h"
#include "../../include/closestHitCommon.h"

layout(location = 1) rayPayloadEXT bool isShadowed;
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
    vec2 xz = vec2(-0.24, -0.22) + sample * vec2(0.47, 0.38);
    return vec3(xz.x, 1.979, xz.y);
}

void main() {
    HitInfo hitInfo = getObjectHitInfo();

    pld.color        = vec3(0.7);
    pld.rayOrigin    = offsetPositionAlongNormal(hitInfo.worldPosition, hitInfo.worldNormal);
    pld.rayDirection = diffuseReflection(hitInfo.worldNormal, pld.rngState);
    pld.rayHitSky    = false;

    vec3 toLight = lightCenter - pld.rayOrigin;
    float distanceToLight = length(toLight);
    // compute BRDF in first pass ?

    const float lightSigma = 1.f;
    const int sqrtShadowRaySamples = 3;
    const float invNSamples = 1.f/sqrtShadowRaySamples;
    vec2 lightSampleSeed;
    for(int i=0; i<sqrtShadowRaySamples; ++i) {
        for(int j=0; j<sqrtShadowRaySamples; ++j) {
            lightSampleSeed.x = invNSamples* (i + stepAndOutputRNGFloat(rngState));
            lightSampleSeed.y = invNSamples* (j + stepAndOutputRNGFloat(rngState));
            vec3 lightSample = sampleAreaLight(lightSampleSeed);
            vec3 center2Sample = lightCenter - lightSample;
            float strength = exp(-0.5 * dot(center2Sample,center2Sample) / (lightSigma * lightSigma))
            vec3 lightDir = normalize(lightSample - pld.rayOrigin);

            if(dot(hitInfo.worldNormal, lightDir) > 0.f) {

            }
        }
    }   

    // Tracing shadow ray only if the light is visible from the surface
    vec3 lightSample = sampleAreaLight(pld.rngState);
    if(dot(hitInfo.worldNormal, lightSample - pld.rayOrigin) > 0) {
        float tMin   = 0.001;
        float tMax   = length(lightSample - pld.rayOrigin);
        vec3  origin = pld.rayOrigin;
        vec3  rayDir = lightSample - pld.rayOrigin;
        rayDir = normalize(rayDir);
        uint  flags =
            gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
        isShadowed = true;
        traceRayEXT(tlas,  // acceleration structure
                flags,       // rayFlags
                0xFF,        // cullMask
                0,           // sbtRecordOffset
                0,           // sbtRecordStride
                1,           // missIndex
                origin,      // ray origin
                tMin,        // ray min range
                rayDir,      // ray direction
                tMax,        // ray max range
                1            // payload (location = 1)
        );

        if(isShadowed) {
            pld.lightCarry = vec3(0.0,0,0);
        } else {
            // Specular
            pld.lightCarry = vec3(0.5) * dot(hitInfo.worldNormal, normalize(lightSample - pld.rayOrigin));
        }
    }
    else {
        pld.lightCarry = vec3(0.0);
    }
}