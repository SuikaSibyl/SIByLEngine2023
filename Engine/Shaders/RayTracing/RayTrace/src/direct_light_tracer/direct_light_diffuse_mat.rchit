#version 460
#extension GL_GOOGLE_include_directive : require

#include "../../../../Utility/random.h"
#include "../../include/closestHitCommon.h"

// The payload:
layout(location = 0) rayPayloadInEXT PassableInfo pld;
layout(location = 1) rayPayloadEXT bool isShadowed;
layout(binding = 0, set = 0) uniform accelerationStructureEXT tlas;

vec3 sampleAreaLight(inout uint rngState) {
    vec2 lsample = vec2(stepAndOutputRNGFloat(rngState), stepAndOutputRNGFloat(rngState));
    vec2 xz = vec2(-0.24, -0.22) + lsample * vec2(0.47, 0.38);
    return vec3(xz.x, 1.8, xz.y);
}

void main() {
    HitInfo hitInfo = getObjectHitInfo();
    pld.color        = vec3(0.7);
    pld.rayOrigin    = offsetPositionAlongNormal(hitInfo.worldPosition, hitInfo.worldNormal);
    pld.rayDirection = diffuseReflection(hitInfo.worldNormal, pld.rngState);
    pld.rayHitSky    = false;

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