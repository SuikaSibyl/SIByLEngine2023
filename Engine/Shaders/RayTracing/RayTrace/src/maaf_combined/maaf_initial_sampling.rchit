#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "maaf_common.h"
#include "../../include/closestHitCommon.h"
#include "../../../../Utility/random.h"
#include "../../../../Utility/sampling.h"
#include "../../../../Utility/geometry.h"

layout(location = 0) rayPayloadInEXT PrimaryRayPayload  primaryPayLoad;
layout(location = 1) rayPayloadEXT   ShadowRayPayload   shadowRayPayLoad;
layout(location = 2) rayPayloadEXT   IndirectRayPayload indirectRayPayLoad;

layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;

void main() {
    // get primary hit info
    HitInfo hitInfo = getObjectHitInfo();
    // set albedo of hit point
    primaryPayLoad.brdf = Kd * getAlbedo(hitInfo.worldNormal);
    primaryPayLoad.worldLocation = hitInfo.worldPosition;
    primaryPayLoad.worldNormal = hitInfo.worldNormal;
    primaryPayLoad.rayHitSky = false;
    // prepare for secondary ray
    const vec3 secondaryRayOrigin = offsetPositionAlongNormal(hitInfo.worldPosition, hitInfo.worldNormal);
    
    // shadow ray
    // ---------------------
    for(int i=0; i<primaryPayLoad.spp_d; ++i) {
        const vec2 lsample = vec2(stepAndOutputRNGFloat(primaryPayLoad.rngState), stepAndOutputRNGFloat(primaryPayLoad.rngState));
        const vec3 lightSample = sampleAreaLight(lsample);
        const vec3 center2Sample = lightSample - lightCenter;
        primaryPayLoad.y0 = center2Sample.x;
        primaryPayLoad.y1 = center2Sample.z;
        const float strength = exp(-0.5 * dot(center2Sample,center2Sample) / (lightSigma * lightSigma));
        const vec3 lightDir = lightSample - secondaryRayOrigin;
        const vec3 toLight = lightCenter - secondaryRayOrigin;
        const float distanceToLight = length(lightDir);
        if(dot(hitInfo.worldNormal, lightDir) > 0.f) {
            // cast shadow ray
            shadowRayPayLoad.hitOccluder = false;
            shadowRayPayLoad.distanceMin = k_inf;
            shadowRayPayLoad.distanceMax = 0;
            // shadowPayLoad.attenuation = vec3(0,0,1);
            traceRayEXT(tlas,   // acceleration structure
                    gl_RayFlagsNoneEXT,       // rayFlags
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
            // if hit any shadow caster
            if(shadowRayPayLoad.hitOccluder) {
                const float d2min = min(max(distanceToLight - shadowRayPayLoad.distanceMax, 0), distanceToLight);
                const float d2max = min(max(distanceToLight - shadowRayPayLoad.distanceMin, 0), distanceToLight);
                const float s1 = distanceToLight/d2min - 1.0;
                const float s2 = distanceToLight/d2max - 1.0;
                primaryPayLoad.direcetSlopeMinMax.x = min(primaryPayLoad.direcetSlopeMinMax.x, s2);
                primaryPayLoad.direcetSlopeMinMax.y = max(primaryPayLoad.direcetSlopeMinMax.y, s1);
            }
            else {
                const float NdL = max(dot(hitInfo.worldNormal, normalize(toLight)),0.0f);
                primaryPayLoad.visibility += NdL;
            }
        }
    }
    if(primaryPayLoad.spp_d > 0)
        primaryPayLoad.visibility /= float(primaryPayLoad.spp_d);
    // indirect ray
    // -------------------------------------
    for(int i=0; i<primaryPayLoad.spp_i; ++i) {
        const int bounceNum = 1;
        // create ONB from world normal
        vec3 u,v,w;
        createONB(hitInfo.worldNormal, u, v, w);
        // create indirect sample
        const vec3 indirectDirSample = hitInfo.worldNormal + randomPointInSphere(primaryPayLoad.rngState);
        vec3 rayOrigin = secondaryRayOrigin;
        vec3 rayDir = normalize(indirectDirSample);
        // get sample v0, v1
        const float vl = 1.f / dot(rayDir, w);
        primaryPayLoad.v0 = dot(rayDir, u) * vl;
        primaryPayLoad.v1 = dot(rayDir, v) * vl;
        vec3 attenuation = vec3(1.);
        vec3 Li = vec3(0.);
        for(int k=0; k<bounceNum; ++k) {
            // cast secondary ray
            indirectRayPayLoad.hit = false;
            indirectRayPayLoad.distanceToReflector = 0;
            indirectRayPayLoad.L = vec3(0.);
            traceRayEXT(tlas,   // acceleration structure
                    gl_RayFlagsOpaqueEXT,       // rayFlags
                    0xFF,        // cullMask
                    2,           // sbtRecordOffset
                    0,           // sbtRecordStride
                    2,           // missIndex
                    rayOrigin,  // ray origin
                    0.0,         // ray min range
                    rayDir,      // ray direction
                    10000.0,        // ray max range
                    2            // payload (location = 1)
            );
            if(indirectRayPayLoad.hit == false)
                break;
            if(k==0) {
                primaryPayLoad.reflectorDistMinMax.x = min(primaryPayLoad.reflectorDistMinMax.x, indirectRayPayLoad.distanceToReflector);
                primaryPayLoad.reflectorDistMinMax.y = max(primaryPayLoad.reflectorDistMinMax.y, indirectRayPayLoad.distanceToReflector);
            }
            rayOrigin = indirectRayPayLoad.worldPosition;
            vec3 tmpNormal = indirectRayPayLoad.worldNormal;
            rayDir = tmpNormal + randomPointInSphere(primaryPayLoad.rngState);
            attenuation *= indirectRayPayLoad.albedo;
            Li += attenuation  * indirectRayPayLoad.L;
        }
        primaryPayLoad.indirect += Li; // if do uniform sample: * secNdL * 2
    }
    if(primaryPayLoad.spp_i > 0)
        primaryPayLoad.indirect /= float(primaryPayLoad.spp_i);
}