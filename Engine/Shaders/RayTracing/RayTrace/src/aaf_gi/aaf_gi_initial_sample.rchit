#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "aaf_gi_common.h"
#include "../../include/closestHitCommon.h"
#include "../../../../Utility/random.h"
#include "../../../../Utility/sampling.h"

layout(location = 0) rayPayloadInEXT PrimarySamplePayload primaryPayLoad;
layout(location = 1) rayPayloadEXT   SecondaryRayPayload  secondaryPayload;
layout(location = 2) rayPayloadEXT   ShadowRayPayload shadowRayPayLoad;

layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;

void main() {
    // get primary hit info
    HitInfo hitInfo = getObjectHitInfo();
    // set albedo of hit point
    primaryPayLoad.albedo = Kd * getAlbedo(hitInfo.worldNormal);
    // prepare for secondary ray
    const vec3 secondaryRayOrigin = offsetPositionAlongNormal(hitInfo.worldPosition, hitInfo.worldNormal);
    const vec3 toLight = lightPos - secondaryRayOrigin;
    // compute direct
    const vec3 L = normalize(toLight);
    const float NdL = max(dot(hitInfo.worldNormal, L),0.0f);
    const vec3 lightDir = lightPos - secondaryRayOrigin;
    traceRayEXT(tlas,           // acceleration structure
        gl_RayFlagsOpaqueEXT,   // rayFlags
        0xFF,        // cullMask
        2,           // sbtRecordOffset
        0,           // sbtRecordStride
        2,           // missIndex
        secondaryRayOrigin,  // ray origin
        0.0,         // ray min range
        normalize(lightDir),      // ray direction
        length(lightDir),         // ray max range
        2               // payload (location = 1)
    );
    if(shadowRayPayLoad.hit == false) {
        primaryPayLoad.rayHitShadow = true;
        primaryPayLoad.direct = NdL * primaryPayLoad.albedo;
    }
    // perpare TBN for secondary rays' normal
    const vec3 seedVec = normalize(vec3(stepAndOutputRNGFloat(primaryPayLoad.rngState),stepAndOutputRNGFloat(primaryPayLoad.rngState),0));
    const vec3 tangent = normalize(seedVec - hitInfo.worldNormal*dot(seedVec, hitInfo.worldNormal));
    const vec3 bitangent = cross(hitInfo.worldNormal, tangent);
    const mat3 TBN = mat3(tangent, bitangent, hitInfo.worldNormal);
    // sample secondary ray
    const int sqrtSecondaryRaySamples = 4;
    const float invNSamples = 1.f/sqrtSecondaryRaySamples;
    const int bounceNum = 1;
    vec2 secondarySampleSeed;
    for(int i=0; i<sqrtSecondaryRaySamples; ++i) {
        for(int j=0; j<sqrtSecondaryRaySamples; ++j) {
            // stratified sampling the hemisphere
            secondarySampleSeed.x = invNSamples* (i + stepAndOutputRNGFloat(primaryPayLoad.rngState));
            secondarySampleSeed.y = invNSamples* (j + stepAndOutputRNGFloat(primaryPayLoad.rngState));
            vec3 sampleDir = cosineSampleHemisphere(secondarySampleSeed);
            const float secNdL = abs(sampleDir.z);
            sampleDir = TBN * sampleDir;
            vec3 rayOrigin = secondaryRayOrigin;
            vec3 rayDir = normalize(sampleDir);
            vec3 attenuation = vec3(1.);
            vec3 Li = vec3(0.);
            for(int k=0; k<bounceNum; ++k) {
                // cast secondary ray
                secondaryPayload.hit = false;
                secondaryPayload.distanceToReflector = 0;
                secondaryPayload.L = vec3(0.);
                traceRayEXT(tlas,   // acceleration structure
                        gl_RayFlagsOpaqueEXT,       // rayFlags
                        0xFF,        // cullMask
                        1,           // sbtRecordOffset
                        0,           // sbtRecordStride
                        1,           // missIndex
                        rayOrigin,  // ray origin
                        0.0,         // ray min range
                        rayDir,      // ray direction
                        10000.0,        // ray max range
                        1            // payload (location = 1)
                );
                if(secondaryPayload.hit == false)
                    break;
                if(k==0) {
                    primaryPayLoad.z_min = min(primaryPayLoad.z_min, secondaryPayload.distanceToReflector);
                    primaryPayLoad.z_max = max(primaryPayLoad.z_max, secondaryPayload.distanceToReflector);
                }
                rayOrigin = secondaryPayload.worldPosition;
                vec3 tmpNormal = secondaryPayload.worldNormal;
                rayDir = tmpNormal + randomPointInSphere(primaryPayLoad.rngState);
                attenuation *= secondaryPayload.albedo;
                Li += attenuation  * secondaryPayload.L;

            }
            primaryPayLoad.indirect += Li; // if do uniform sample: * secNdL * 2
        }
    }

    primaryPayLoad.indirect /= sqrtSecondaryRaySamples * sqrtSecondaryRaySamples;

    // write to primary ray payload
    primaryPayLoad.rayHitSky = false;
    primaryPayLoad.hitPoint = hitInfo.worldPosition;
    primaryPayLoad.hitNormal = vec3(hitInfo.uv, 0);
}