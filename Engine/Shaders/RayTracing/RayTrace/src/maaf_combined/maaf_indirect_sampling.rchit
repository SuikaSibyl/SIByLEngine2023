#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "maaf_common.h"
#include "../../include/closestHitCommon.h"

layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;

layout(location = 2) rayPayloadInEXT IndirectRayPayload indirectRayPayLoad;
layout(location = 3) rayPayloadEXT   bool shadowHit;

void main() {
    // get primary hit info
    HitInfo hitInfo = getObjectHitInfo();
    // Compute the distance
    const float tHit = length(hitInfo.worldPosition - gl_WorldRayOriginEXT);
    // shadow ray start pos
    const vec3 shadowRayOrigin = offsetPositionAlongNormal(hitInfo.worldPosition, hitInfo.worldNormal);
    // Fill payload
    indirectRayPayLoad.L = vec3(0.f);
    indirectRayPayLoad.hit = true;
    indirectRayPayLoad.distanceToReflector = tHit;
    indirectRayPayLoad.worldPosition = shadowRayOrigin;
    indirectRayPayLoad.worldNormal = hitInfo.worldNormal;
    indirectRayPayLoad.albedo = Kd * getAlbedo(hitInfo.worldNormal);
    // measrue Lo term
    shadowHit = false;
    const vec3 lightDir = lightCenter - shadowRayOrigin;
    traceRayEXT(tlas,           // acceleration structure
        gl_RayFlagsOpaqueEXT,   // rayFlags
        0xFF,                   // cullMask
        3,                      // sbtRecordOffset
        0,                      // sbtRecordStride
        3,                      // missIndex
        shadowRayOrigin,        // ray origin
        0.0,                    // ray min range
        normalize(lightDir),    // ray direction
        length(lightDir),       // ray max range
        3                       // payload (location = 3)
    );
    if(shadowHit == false) {
        // compute BRDF
        const vec3 L = normalize(lightDir);
        const float NdL = max(dot(hitInfo.worldNormal, L),0.0f);
        indirectRayPayLoad.L = NdL * vec3(1);
    }
}