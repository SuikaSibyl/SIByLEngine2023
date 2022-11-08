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
    secondaryPayLoad.hit = true;
    secondaryPayLoad.distanceToReflector = tHit;
    secondaryPayLoad.worldPosition = shadowRayOrigin;
    secondaryPayLoad.worldNormal = hitInfo.worldNormal;
    secondaryPayLoad.albedo = Kd * getAlbedo(hitInfo.worldNormal);
    // measrue Lo term
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
        secondaryPayLoad.L = NdL * vec3(1);
    }
}