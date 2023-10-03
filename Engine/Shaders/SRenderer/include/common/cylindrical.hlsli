#ifndef _SRENDERER_CYLINDRICAL_HEADER_
#define _SRENDERER_CYLINDRICAL_HEADER_

#include "cpp_compatible.hlsli"
#include "math.hlsli"

/**
 * Converts a uniform vector to cylindrical coordinates.
 * It is the Lambert cylindrical equal-area projection, and is area-preserving.
 * @param unitVector The unit vector to convert.
 * @return The cylindrical coordinates: (cos\theta, \phi).
 * output.x: cos\theta: [-1, 1].
 * output.y: \phi: [-\pi, \pi].
 */
float2 UnitVectorToCylindrical(in_ref(float3) unitVector) {
    const float cosTheta = unitVector.z;
    const float phi = atan2(unitVector.y, unitVector.x);
    return float2(cosTheta, phi);
}

/**
 * Converts a cylindrical vector to a unit vector.
 * @param cylindrical The cylindrical coordinates: (cos\theta, \phi).
 * @return The unit vector in [0,1]^2.
 */
float2 CylindricalToUnit(in_ref(float2) cylindrical) {
    return saturate((cylindrical + float2(1.0, k_pi)) / float2(2.0, 2.0 * k_pi));
}

/**
 * Converts a unit vector to cylindrical vector.
 * @param uniform The uniform coordinates: [-1, 1].
 * @return The cylindrical coordinates.
 * output.x: cos\theta: [-1, 1].
 * output.y: \phi: [-\pi, \pi].
 */
float2 UnitToCylindrical(in_ref(float2) uniform) {
    return clamp((uniform * float2(2.0, 2.0 * k_pi)) - float2(1.0, k_pi),
                 float2(-1.0, -k_pi), float2(1.0, k_pi));
}

/**
 * Converts a cylindrical coordinates to uniform vector.
 * @param unitVector The unit vector to convert.
 * @return unitVector The unit vector converted.
 */
float3 CylindricalToUnitVector(in_ref(float2) cylindrical) {
    const float cosTheta = cylindrical.x;
    const float phi = cylindrical.y;
    const float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    float3 unitVector;
    unitVector.x = cos(phi) * sinTheta;
    unitVector.y = sin(phi) * sinTheta;
    unitVector.z = cosTheta;
    return normalize(unitVector);
}

#endif // _SRENDERER_CYLINDRICAL_HEADER_