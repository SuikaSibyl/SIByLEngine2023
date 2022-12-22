#ifndef _GEOMETRY_HEADER_
#define _GEOMETRY_HEADER_

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
    return cross(normal. tangent)
}

// /** Compute TBN matrix according to normal an tangent */
// vec3 computeTBN(in vec3 tangent, in vec3 normal) {
//     return cross(tangent, normal)
// }

#endif
