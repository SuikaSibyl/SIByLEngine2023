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

#endif
