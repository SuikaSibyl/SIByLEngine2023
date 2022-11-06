#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "lwb_common.h"
#include "../../include/closestHitCommon.h"
#include "../../../../Utility/random.h"
#include "../../../../Utility/sampling.h"

layout(location = 0) rayPayloadInEXT PrimaryRayPayload primaryPayLoad;
// layout(location = 1) rayPayloadEXT   SecondaryRayPayload  secondaryPayload;
// layout(location = 2) rayPayloadEXT   ShadowRayPayload shadowRayPayLoad;

layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;

void main() {
    // get primary hit info
    HitInfo hitInfo = getObjectHitInfo();
    primaryPayLoad.hitPoint = hitInfo.worldPosition;
    primaryPayLoad.hitNormal = hitInfo.worldNormal;
    primaryPayLoad.color = vec3(0.5);
    // const vec3 secondaryRayOrigin = offsetPositionAlongNormal(hitInfo.worldPosition, hitInfo.worldNormal);

    // const vec3 toLight = lightPos - secondaryRayOrigin;

    // // compute BRDF
    // if(true) {
    //     const vec3 L = normalize(toLight);
    //     const float NdL = max(dot(hitInfo.worldNormal, L),0.0f);
    //     const vec3 H = normalize(L - normalize(gl_WorldRayDirectionEXT));
    //     const float NdH = max(dot(hitInfo.worldNormal, H),0.0f);
    //     vec3 color = vec3(0);
    //     // white light
    //     const vec3 Kd = vec3(1.);
    //     color += Kd * NdL;
    //     // do not ocnsider specular now
    //     // if (NdH > 0)
    //     //     color += Ks * pow(NdH, phongExp);
    //     primaryPayLoad.brdf = color * getAlbedo(hitInfo.worldNormal);
    // }

    // const vec3 lightDir = lightPos - secondaryRayOrigin;
    // traceRayEXT(tlas,           // acceleration structure
    //     gl_RayFlagsOpaqueEXT,   // rayFlags
    //     0xFF,        // cullMask
    //     2,           // sbtRecordOffset
    //     0,           // sbtRecordStride
    //     2,           // missIndex
    //     secondaryRayOrigin,  // ray origin
    //     0.0,         // ray min range
    //     normalize(lightDir),      // ray direction
    //     length(lightDir),         // ray max range
    //     2               // payload (location = 1)
    // );
    // if(shadowRayPayLoad.hit == false) {
    //     primaryPayLoad.rayHitShadow = true;
    // }

    // // secondaryPayload.hit = false;
    // // secondaryPayload.attenuation = vec3(1,0,0);
    // // secondaryPayload.distanceMin = 9999;
    // // secondaryPayload.distanceMax = 0;
    // const vec3 seedVec = normalize(vec3(primaryPayLoad.rngState,primaryPayLoad.rngState,0));
    // const vec3 tangent = normalize(seedVec - hitInfo.worldNormal*dot(seedVec, hitInfo.worldNormal));
    // const vec3 bitangent = cross(hitInfo.worldNormal, tangent);
    // const mat3 TBN = mat3(tangent, bitangent, hitInfo.worldNormal);

    // const int sqrtSecondaryRaySamples = 4;
    // const float invNSamples = 1.f/sqrtSecondaryRaySamples;
    // vec2 secondarySampleSeed;
    // for(int i=0; i<sqrtSecondaryRaySamples; ++i) {
    //     for(int j=0; j<sqrtSecondaryRaySamples; ++j) {
    //         // stratified sampling the hemisphere
    //         secondarySampleSeed.x = invNSamples* (i + stepAndOutputRNGFloat(primaryPayLoad.rngState));
    //         secondarySampleSeed.y = invNSamples* (j + stepAndOutputRNGFloat(primaryPayLoad.rngState));
    //         vec3 sampleDir = uniformSampleHemisphere(secondarySampleSeed);
    //         // sampleDir.z = abs(sampleDir.z);
    //         sampleDir = TBN * sampleDir;
    //         // sampleDir =  vec3(sampleDir.x, sampleDir.z, sampleDir.y);
    //         // vec3 sampleDir = uniformSampleHemisphere(secondarySampleSeed);
    //         // sampleDir = TBN * sampleDir;
    //         // if(primaryPayLoad.rngState > 0.5) sampleDir.z = -sampleDir.z;
    //         // sampleDir = hitInfo.worldNormal;
    //         // const vec3 sampleDir = hitInfo.worldNormal + randomPointInSphere(primaryPayLoad.rngState);
    // //         const vec3 center2Sample = lightCenter - lightSample;
    // //         const float strength = exp(-0.5 * dot(center2Sample,center2Sample) / (lightSigma * lightSigma));
    // //         const vec3 lightDir = lightSample - secondaryRayOrigin;
    //         const float strength = 1;

    //         // accumulate weights for vis
    //         primaryPayLoad.accumWeights += strength;
    //         // cast shadow ray
    //         secondaryPayload.hit = false;
    //         secondaryPayload.distanceToReflector = 0;
    //         secondaryPayload.L = vec3(0.);
    //         // shadowPayLoad.attenuation = vec3(0,0,1);
    //         uint  flags = gl_RayFlagsOpaqueEXT;
    //         traceRayEXT(tlas,   // acceleration structure
    //                 flags,       // rayFlags
    //                 0xFF,        // cullMask
    //                 1,           // sbtRecordOffset
    //                 0,           // sbtRecordStride
    //                 1,           // missIndex
    //                 secondaryRayOrigin,  // ray origin
    //                 0.0,         // ray min range
    //                 normalize(sampleDir),      // ray direction
    //                 10000.0,        // ray max range
    //                 1            // payload (location = 1)
    //         );
    //         // if hit any shadow caster
    //         if(secondaryPayload.hit) {
    //             // primaryPayLoad.rayShadow = true;
    //             // float d2min = distanceToLight - shadowPayLoad.distanceMax;
    //             // const float d2max = distanceToLight - shadowPayLoad.distanceMin;
    //             // if (shadowPayLoad.distanceMax < 0.000000001)
    //             //     d2min = distanceToLight;
    //             // const float s1 = distanceToLight/d2min - 1.0;
    //             // const float s2 = distanceToLight/d2max - 1.0;
    //             // primaryPayLoad.s1 = max(primaryPayLoad.s1, s1);
    //             // primaryPayLoad.s2 = min(primaryPayLoad.s2, s2);
    //         }
    //         // if shadow ray hit no objects
    //         else {
    //             const vec3 L = normalize(sampleDir);
    //             const float NdL = max(dot(hitInfo.worldNormal, L),0.0f);
    //             vec3 color = vec3(0);
    //             // white light
    //             const vec3 Kd = vec3(1.);
    //             color += getAlbedo(hitInfo.worldNormal) * NdL;

    //             primaryPayLoad.accumGI += secondaryPayload.L * color;
    //         }
    //     }
    // }

    // if(primaryPayLoad.accumWeights > 0)
    //     primaryPayLoad.accumGI /= primaryPayLoad.accumWeights;
    // // primaryPayLoad.accumGI = getAlbedo(hitInfo.worldNormal) * primaryPayLoad.brdf;
    // // // write to primary ray payload
    // // primaryPayLoad.rayHitSky = false;
    // // primaryPayLoad.hitPoint = hitInfo.worldPosition;
    // // primaryPayLoad.hitNormal = hitInfo.worldNormal;
}