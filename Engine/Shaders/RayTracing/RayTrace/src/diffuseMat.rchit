#version 460
#extension GL_GOOGLE_include_directive : require

#include "../include/closestHitCommon.h"

void main() {
    HitInfo hitInfo = getObjectHitInfo();
    pld.color        = vec3(0.7);
    pld.rayOrigin    = offsetPositionAlongNormal(hitInfo.worldPosition, hitInfo.worldNormal);
    pld.rayDirection = diffuseReflection(hitInfo.worldNormal, pld.rngState);
    pld.rayHitSky    = false;
}
