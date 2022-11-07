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
layout(location = 1) rayPayloadEXT bool shadowHit;
layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;

const float phongExp = 100;

void main() {
    // get primary hit info
    HitInfo hitInfo = getObjectHitInfo();
    const vec3 secondaryRayOrigin = offsetPositionAlongNormal(hitInfo.worldPosition, hitInfo.worldNormal);

    int sampleCount = int(primaryPayLoad.accumWeights);
    primaryPayLoad.accumWeights = 0.f;

    for(int i=0; i<sampleCount; ++i) {
        // stratified sampling the light
        const vec3 lightSample = sampleAreaLight(primaryPayLoad.rngState);
        const vec3 center2Sample = lightCenter - lightSample;
        const float strength = exp(-0.5 * dot(center2Sample,center2Sample) / (lightSigma * lightSigma));
        const vec3 lightDir = lightSample - secondaryRayOrigin;

        if(dot(hitInfo.worldNormal, lightDir) > 0.f) {
            // accumulate weights for vis
            primaryPayLoad.accumWeights += strength;
            // cast shadow ray
            shadowHit = false;
            // shadowPayLoad.attenuation = vec3(0,0,1);
            traceRayEXT(tlas,   // acceleration structure
                    gl_RayFlagsOpaqueEXT,       // rayFlags
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
            if(!shadowHit) {
                primaryPayLoad.accumVis += strength;
            }
        }
    }
    // write to primary ray payload
    primaryPayLoad.rayHitSky = false;
    primaryPayLoad.hitPoint = hitInfo.worldPosition;
    primaryPayLoad.hitNormal = hitInfo.worldNormal;
}