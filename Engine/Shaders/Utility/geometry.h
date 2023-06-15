#ifndef _GEOMETRY_HEADER_
#define _GEOMETRY_HEADER_

#include "math.h"

/**
* Create ONB (orthonormal bases bases) from normal.
* Resulting w is parallel to normal.
*/
void createONB(in vec3 n, out vec3 u, out vec3 v, out vec3 w) {
    w = normalize(n);
    u = cross(w, vec3(0.0f, 1.0f, 0.0f));
    if (abs(u.x)<0.001f && abs(u.y)<0.001f && abs(u.z)<0.001f)
        u = cross(w, vec3(1.0f, 0.0f, 0.0f));
    u = normalize(u);
    v = cross(w,u);
}

/** Compute bitangent according to tangent and normal */
vec3 computeBitangent(in vec3 tangent, in vec3 normal) {
    return cross(normal, tangent);
}

/**
* Create ONB (orthonormal bases bases) from normal.
* Resulting w is parallel to normal.
*/
mat3 createONB(in vec3 n) {
    vec3 w = normalize(n);
    vec3 u = cross(w, vec3(0.0f, 1.0f, 0.0f));
    if (abs(u.x)<0.001f && abs(u.y)<0.001f && abs(u.z)<0.001f)
        u = cross(w, vec3(1.0f, 0.0f, 0.0f));
    u = normalize(u);
    vec3 v = cross(w,u);
    return mat3(u,v,w);
}

mat3 buildTangentToWorld(in vec4 tangentWS, in vec3 normalWS) {
    // Gram-Schmidt process
    vec3 T = normalize(tangentWS.xyz);
    vec3 N = normalize(normalWS);
    // re-orthogonalize T with respect to N
    T = normalize(T - dot(T, N) * N);
    // then retrieve perpendicular vector B with the cross product of T and N
    vec3 B = cross(N, T) * tangentWS.w;
    return mat3(T, B, N);
}

// /** Compute TBN matrix according to normal */
mat3 createFrame(in vec3 n) {
    if (n[2] < float(-1 + 1e-6)) {
        return mat3(vec3(0, -1, 0),
                    vec3(-1, 0, 0),
                    n);
    } else {
        const float a = 1 / (1 + n[2]);
        const float b = -n[0] * n[1] * a;
        return mat3(vec3(1 - n[0] * n[0] * a, b, -n[0]),
                    vec3(b, 1 - n[1] * n[1] * a, -n[1]),
                    n);
    }
}

/************************************************************************
*                           Bounding Box Utils                          *
************************************************************************/
struct AABB {
    vec3 min;
    vec3 max;
};

AABB aabbUnion(in AABB a, in AABB b) {
    AABB result;
    result.min = min(a.min, b.min);
    result.max = max(a.max, b.max);
    return result;
}

/************************************************************************
*                            Rotation Utils                             *
************************************************************************/

/** Return a matrix that rotate vector around axis by angle. */
mat4 rotationMatrix(
    in const vec3 axis,     // should be normalized
    in const float angle
) {
    const float s = sin(angle);
    const float c = cos(angle);
    const float oc = 1.0 - c;
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

/** Rotate vector around axis by angle */
vec3 rotate(
    in const vec3 v,
    in const vec3 axis,     // should be normalized
    in const float angle
) {
    return (rotationMatrix(axis, angle) * vec4(v, 0.0)).xyz;
}

/************************************************************************
*                             Normal Utils                              *
************************************************************************/

// Normal Cone
// -----------------------------------------------------------------------

struct NormalCone {
    vec3 direction;
    float cosAngle;
};

struct NormalConeEx {
    vec3 direction;
    float theta_e;
    float theta_o;
    bool  valid;
};

NormalConeEx coneUnion(
    in NormalConeEx a,
    in NormalConeEx b
) {
    if(!a.valid && !b.valid) return NormalConeEx(
        vec3(0.0f, 0.0f, 0.0f),
        0.0f,
        0.0f,
        false
    );
    if(!a.valid) return b;
    if(!b.valid) return a;

    NormalConeEx result;
    result.valid = true;
    if (b.theta_o > a.theta_o) {
        // Swap(a, b);
        NormalConeEx tmp = a; a = b; b = tmp;
    }
    const float theta_d = acos(dot(a.direction, b.direction));
    const float theta_e = max(a.theta_e, b.theta_e);
    if (min(theta_d + b.theta_o, k_pi) <= a.theta_o) {
        result.direction = a.direction;
        result.theta_o = a.theta_o;
        result.theta_e = theta_e;
        return result;
    }
    else {
        const float theta_o = (theta_d + a.theta_o + b.theta_o) * 0.5f;
        if(k_pi <= theta_o) {
            return NormalConeEx(a.direction, k_pi, theta_e, true);
        }
        const float theta_r = theta_o - a.theta_o;
        const vec3 direction = rotate(a.direction, normalize(cross(a.direction, b.direction)), -theta_r);
        if(isnan(direction.x)) {
            return NormalConeEx(a.direction, theta_o, theta_e, true);
        }
        return NormalConeEx(normalize(direction), theta_o, theta_e, true);
    }
}

vec3 packNormal(in const vec3 normal) {
    return 0.5 * vec3(normal + 1.0);
}

vec3 unpackNormal(in const vec3 normal) {
    return normalize(normal * 2.0 - 1.0);
}

// vec2 UnitVectorToOctahedron(in vec3 N) {
//     N.xy /= dot(vec3(1), abs(N) );
//     if( N.z <= 0 ) {
//         N.xy = (1 - abs(N.yx) ) * ( N.xy >= 0 ? vec2(1,1) : vec2(-1,-1) );
//     }
//     return N.xy;
// }

// vec3 OctahedronToUnitVector(in const vec2 Oct) {
//     const vec3 N = float3( Oct, 1 - dot( 1, abs(Oct) ) );
//     if( N.z < 0 ) {
//         N.xy = ( 1 - abs(N.yx) ) * ( N.xy >= 0 ? vec2(1,1) : vec2(-1,-1) );
//     }
//     return normalize(N);
// }

/************************************************************************
*                             Intersection                              *
************************************************************************/

/** Light weight ray triangle intersection
* @ref: https://github.com/SebLague/Gamedev-Maths/blob/master/PointInTriangle.cs */
bool ray_triangle_intersection_lightweight(
    in const vec3 ray, in const vec3 rayDir,
    in const vec3 A, in const vec3  B, in const  vec3 C,
    out vec2 barycentricCoord
) {
    vec3 normal = normalize(cross(B - A, C - A));
    float t = dot(-ray, normal) / dot(normal, rayDir);
    vec3 P = ray + rayDir * t;  // hit point

    float s1 = C.y - A.y;
    float s2 = C.x - A.x;
    float s3 = B.y - A.y;
    float s4 = P.y - A.y;

    float w1 = (A.x * s1 + s4 * s2 - P.x * s1) / (s3 * s2 - (B.x - A.x) * s1);
    float w2 = (s4 - w1 * s3) / s1;

    barycentricCoord = vec2(w1, w2);
    return w1 >= 0.0 && w2 >= 0.0 && (w1 + w2) <= 1.0 ? true : false;
}

const float MachineEpsilon = 1.0e-6f;

float gamma(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

bool ray_triangle_intersection_pbrt(
    in const vec3 ro, in const vec3 rd, in const float tmin, in const float tmax,
    in const vec3 p0, in const vec3 p1, in const vec3 p2,
    out vec3 barycentricCoord
) {
    // Perform ray–triangle intersection test
    // ** Translate vertices based on ray origin
    vec3 p0t = p0 - ro;
    vec3 p1t = p1 - ro;
    vec3 p2t = p2 - ro;
    // ** Permute components of triangle vertices and ray direction
    int kz = maxDimension(abs(rd));
    int kx = kz + 1; if (kx == 3) kx = 0;
    int ky = kx + 1; if (ky == 3) ky = 0;
    vec3 d = permute(rd, kx, ky, kz);
    p0t = permute(p0t, kx, ky, kz);
    p1t = permute(p1t, kx, ky, kz);
    p2t = permute(p2t, kx, ky, kz);
    // ** Apply shear transformation to translated vertex positions
    float Sx = -d.x / d.z;
    float Sy = -d.y / d.z;
    float Sz = 1.f / d.z;
    p0t.x += Sx * p0t.z;
    p0t.y += Sy * p0t.z;
    p1t.x += Sx * p1t.z;
    p1t.y += Sy * p1t.z;
    p2t.x += Sx * p2t.z;
    p2t.y += Sy * p2t.z;
    // Compute edge function coefficients e0, e1, and e2>> 
    float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
    float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
    float e2 = p0t.x * p1t.y - p0t.y * p1t.x;
    // Fall back to double-precision test at triangle edges
    if (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f) {
        double p2txp1ty = double(p2t.x) * double(p1t.y);
        double p2typ1tx = double(p2t.y) * double(p1t.x);
        // e0 = float(p2typ1tx - p2txp1ty);
        // double p0txp2ty = (double)p0t.x * (double)p2t.y;
        // double p0typ2tx = (double)p0t.y * (double)p2t.x;
        // e1 = float(p0typ2tx - p0txp2ty);
        // double p1txp0ty = (double)p1t.x * (double)p0t.y;
        // double p1typ0tx = (double)p1t.y * (double)p0t.x;
        // e2 = float(p1typ0tx - p1txp0ty);
    }
    // Perform triangle edge and determinant tests 
    if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
        return false;
    float det = e0 + e1 + e2;
    if (det == 0)
        return false;
    // Compute scaled hit distance to triangle and test against ray  range>> 
    p0t.z *= Sz;
    p1t.z *= Sz;
    p2t.z *= Sz;
    float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
    if (det < 0 && (tScaled >= 0 || tScaled < tmax * det))
        return false;
    else if (det > 0 && (tScaled <= 0 || tScaled > tmax * det))
        return false;

    // Compute barycentric coordinates and  value for triangle intersection
    float invDet = 1 / det;
    float b0 = e0 * invDet;
    float b1 = e1 * invDet;
    float b2 = e2 * invDet;
    float t = tScaled * invDet;

    // // Ensure that computed triangle is conservatively greater than zero
    // // ** Compute  term for triangle error bounds
    // float maxZt = maxComponent(abs(vec3(p0t.z, p1t.z, p2t.z)));
    // float deltaZ = gamma(3) * maxZt;
    // // ** Compute  and  terms for triangle  error bounds
    // float maxXt = maxComponent(abs(vec3(p0t.x, p1t.x, p2t.x)));
    // float maxYt = maxComponent(abs(vec3(p0t.y, p1t.y, p2t.y)));
    // float deltaX = gamma(5) * (maxXt + maxZt);
    // float deltaY = gamma(5) * (maxYt + maxZt);
    // // ** Compute  term for triangle  error bounds
    // float deltaE = 2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt +
    //                         deltaX * maxYt);
    // // ** Compute  term for triangle  error bounds and check t
    // float maxE = maxComponent(abs(vec3(e0, e1, e2)));
    // float deltaT = 3 * (gamma(3) * maxE * maxZt + deltaE * maxZt +
    //                         deltaZ * maxE) * abs(invDet);
    // if (t <= deltaT)
    //     return false;

    // Interpolate  parametric coordinates and hit point
    vec3 pHit = b0 * p0 + b1 * p1 + b2 * p2;
    barycentricCoord = vec3(b0, b1, b2);
    // vec3 uvHit = b0 * uv[0] + b1 * uv[1] + b2 * uv[2];
    return true;
}

#endif
