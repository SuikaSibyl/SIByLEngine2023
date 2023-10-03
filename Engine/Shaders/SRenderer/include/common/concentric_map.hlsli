#ifndef _SRNEDERER_CONCENTRIC_MAP_HEADER_
#define _SRNEDERER_CONCENTRIC_MAP_HEADER_

#include "cpp_compatible.hlsli"
#include "math.hlsli"

// /**
//  * Maps a square to a disk .
//  * Adapted from "A Low Distortion Map Between Disk and Square." Improved based
//  * on comments from Peter Shirley's blog post: "Improved code for concentric map."
//  * Adapted from Unreal Engine 4's implementation of `ToConcentricMap`
//  * @file: Engine/Shaders/Private/RayTracing/RayTracingDirectionalLight.ush
//  * @param p: A point on the unit square
//  * @return: A point on the unit disk
//  */
// float2 ToConcentricMap(in_ref(float2) RectangularCoords) {
//     float R; float Phi;
//     RectangularCoords = 2.0 * RectangularCoords - 1.0;
//     float2 RectangularCoordsSquared = RectangularCoords * RectangularCoords;
//     if (RectangularCoordsSquared.x > RectangularCoordsSquared.y) {
//         R = RectangularCoords.x;
//         Phi = (k_pi / 4.0) * (RectangularCoords.y / RectangularCoords.x);
//     } else {
//         R = RectangularCoords.y;
//         Phi = (k_pi / 2.0) - (k_pi / 4.0) * (RectangularCoords.x / RectangularCoords.y);
//     }
//     float2 PolarCoords = float2(R, Phi);
//     return PolarCoords;
// }

// /**
//  * Inverse of `ToConcentricMap`, mapping a disk to a square.
//  * Adapted from Unreal Engine 4's implementation of `InverseConcentricMapping`
//  * @file: Engine/Shaders/Private/PathTracing/Material/PathTracingRadianceProbe.ush
//  * @param p: A point on the unit disk
//  * @return: A point on the unit square
//  */
// float2 FromConcentricMap(in_ref(float2) p) {
//     // Handle degeneracy at the origin
//     if (p.x == 0 && p.y == 0) return float2(0, 0);
//     float r = sqrt(p.x * p.x + p.y * p.y);
//     float theta = atan2(p.y, p.x);
//     if (theta < -k_pi / 4) theta += 2 * k_pi;
//     float a; float b;
//     if (theta < k_pi / 4) { // region 1
//         a = r; b = theta * a / (k_pi / 4);
//     } else if (theta < 3 * k_pi / 4) { // region 2
//         b = r; a = -(theta - k_pi / 2) * b / (k_pi / 4);
//     } else if (theta < 5 * k_pi / 4) { // region 3
//         a = -r; b = (theta - k_pi) * a / (k_pi / 4);
//     } else { // region 4
//         b = -r; a = -(theta - 3 * k_pi / 2) * b / (k_pi / 4);
//     }
//     const float x = (a + 1) / 2;
//     const float y = (b + 1) / 2;
//     return float2(x, y);
// }

float2 ToConcentricMap(in_ref(float2) onSquare) {
    float phi; float r;
    // (a,b) is now on [-1,1]^2
    const float a = 2 * onSquare.x - 1;
    const float b = 2 * onSquare.y - 1;
    if (a > -b) { // region 1 or 2
        if (a > b) { // region 1, also |a| > |b|
            r = a;
            phi = (k_pi / 4) * (b / a);
        }
        else { // region 2, also |b| > |a|
            r = b;
            phi = (k_pi / 4) * (2 - (a / b));
        }
    }
    else { // region 3 or 4
        if (a < b) { // region 3, also |a| > |b|, a!= 0
            r = -a;
            phi = (k_pi / 4) * (4 + (b / a));
        }
        else { // region 4, |b| >= |a|, but a==0 and b==0 could occur.
            r = -b;
            if (b != 0) phi = (k_pi / 4) * (6 - (a / b));
            else phi = 0;
        }
    }
    float u = r * cos(phi);
    float v = r * sin(phi);
    return float2(u, v);
}

float2 FromConcentricMap(in_ref(float2) onDisk) {
    const float r = sqrt(onDisk.x * onDisk.x + onDisk.y * onDisk.y);
    float phi = atan2(onDisk.y, onDisk.x);
    if (phi < -k_pi / 4)
        phi += 2 * k_pi; // in range [-pi/4,7pi/4]
    float a; float b;
    if (phi < k_pi / 4) { // region 1
        a = r;
        b = phi * a / (k_pi / 4);
    }
    else if (phi < 3 * k_pi / 4) { // region 2
        b = r;
        a = -(phi - k_pi / 2) * b / (k_pi / 4);
    }
    else if (phi < 5 * k_pi / 4) { // region 3
        a = -r;
        b = (phi - k_pi) * a / (k_pi / 4);
    }
    else { // region 4
        b = -r;
        a = -(phi - 3 * k_pi / 2) * b / (k_pi / 4);
    }
    const float x = (a + 1) / 2;
    const float y = (b + 1) / 2;
    return float2(x, y);
}

/**
 * Maps a point on the disk to a point on the unit hemisphere.
 * @ref: "A Low Distortion Map Between Disk and Square" - Peter Shirley & Kenneth Chiu
 */
float3 ConcentricDiskToUniformHemisphere(in_ref(float2) onDisk) {
    const float r2 = onDisk.x * onDisk.x + onDisk.y * onDisk.y;
    const float r = sqrt(r2);
    const float z = 1 - r2;
    const float z2 = z * z;
    const float tmp = sqrt(1 - z2);
    const float x = onDisk.x * tmp / r;
    const float y = onDisk.y * tmp / r;
    return float3(x, y, z);
}

/**
 * Maps a point on the unit hemisphere to a point on the disk.
 * @ref: "A Low Distortion Map Between Disk and Square" - Peter Shirley & Kenneth Chiu
 */
float2 UniformHemisphereToConcentricDisk(in_ref(float3) onHemisphere) {
    const float r = sqrt(1 - onHemisphere.z);
    const float tmp = sqrt(1 - onHemisphere.z * onHemisphere.z);
    const float x = onHemisphere.x * r / tmp;
    const float y = onHemisphere.y * r / tmp;
    return float2(x, y);
}

#endif  // _SRNEDERER_CONCENTRIC_MAP_HEADER_