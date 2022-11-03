#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "aaf_gi_common.h"
#include "../../include/closestHitCommon.h"

layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;

layout(location = 1) rayPayloadInEXT SecondaryRayPayload secondaryPayLoad;
layout(location = 2) rayPayloadEXT   ShadowRayPayload shadowRayPayLoad;

void main() {
    // get primary hit info
    HitInfo hitInfo = getObjectHitInfo();
    // Compute the distance
    const float tHit = length(hitInfo.worldPosition - gl_WorldRayOriginEXT);
    // shadow ray start pos
    const vec3 shadowRayOrigin = offsetPositionAlongNormal(hitInfo.worldPosition, hitInfo.worldNormal);

    // Fill payload
    secondaryPayLoad.L = vec3(0.f);
    shadowRayPayLoad.hit = false;

    const vec3 lightDir = lightPos - shadowRayOrigin;

    traceRayEXT(tlas,           // acceleration structure
        gl_RayFlagsOpaqueEXT,   // rayFlags
        0xFF,        // cullMask
        2,           // sbtRecordOffset
        0,           // sbtRecordStride
        2,           // missIndex
        shadowRayOrigin,  // ray origin
        0.0,         // ray min range
        normalize(lightDir),      // ray direction
        length(lightDir),         // ray max range
        2               // payload (location = 1)
    );
    if(shadowRayPayLoad.hit == false) {
        // compute BRDF
        const vec3 L = normalize(lightDir);
        const float NdL = max(dot(hitInfo.worldNormal, L),0.0f);
        const vec3 H = normalize(L - normalize(gl_WorldRayDirectionEXT));
        const float NdH = max(dot(hitInfo.worldNormal, H),0.0f);
        vec3 color = vec3(0);
        // white light
        const vec3 Kd = vec3(1.);
        color += Kd * NdL;
        // do not ocnsider specular now
        // if (NdH > 0)
        //     color += Ks * pow(NdH, phongExp);
        secondaryPayLoad.L = color * getAlbedo(hitInfo.worldNormal);
    }
    // shadowRayPayLoad.hit           = true;
    // shadowPayLoad.attenuation   = vec3(0.f);
    // shadowPayLoad.distanceMin   = min(shadowPayLoad.distanceMin, tHit);
    // shadowPayLoad.distanceMax   = max(shadowPayLoad.distanceMax, tHit);

    
}