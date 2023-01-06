#ifndef _SRENDERER_COMMON_HIT_HEADER_
#define _SRENDERER_COMMON_HIT_HEADER_

#include "common_trace.h"

// This will store two of the barycentric coordinates of the intersection when
// closest-hit shaders are called:
hitAttributeEXT vec2 attributes;

// Hit geometry data to return
struct HitGeometry {
    vec3 worldPosition;
    uint matID;
    vec2 uv;
    mat3 TBN;
};

HitGeometry getHitGeometry() {
    HitGeometry hit;
    // Get all ids
    const int primitiveID = gl_PrimitiveID;
    const int geometryID = gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT;
    const GeometryInfo geometryInfo = geometryInfos[geometryID];
    // Get the indices of the vertices of the triangle
    const uint i0 = indices[3 * primitiveID + 0 + geometryInfo.indexOffset];
    const uint i1 = indices[3 * primitiveID + 1 + geometryInfo.indexOffset];
    const uint i2 = indices[3 * primitiveID + 2 + geometryInfo.indexOffset];
    // Get the vertices of the triangle
    mat4 o2w = ObjectToWorld(geometryInfo);
    const vec3 v0 = vertices[i0 + geometryInfo.vertexOffset].position;
    const vec3 v1 = vertices[i1 + geometryInfo.vertexOffset].position;
    const vec3 v2 = vertices[i2 + geometryInfo.vertexOffset].position;
    // Get the barycentric coordinates of the intersection
    vec3 barycentrics = vec3(0.0, attributes.x, attributes.y);
    barycentrics.x    = 1.0 - barycentrics.y - barycentrics.z;
    // Get matID
    hit.matID = geometryInfo.materialID;
    // Get position
    vec3 position = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
    vec4 positionWorld =  o2w * vec4(position, 1);
    hit.worldPosition = positionWorld.xyz;
    // Get texcoord
    const vec2 u0 = vertices[i0 + geometryInfo.vertexOffset].texCoords;
    const vec2 u1 = vertices[i1 + geometryInfo.vertexOffset].texCoords;
    const vec2 u2 = vertices[i2 + geometryInfo.vertexOffset].texCoords;
    hit.uv = u0 * barycentrics.x + u1 * barycentrics.y + u2 * barycentrics.z;
    // Get BTN
    const mat4 o2wn = ObjectToWorldNormal(geometryInfo);
    vec3 n[3], t[3];
    mat3 TBN[3];
    n[0] = vertices[i0 + geometryInfo.vertexOffset].normal;
    n[1] = vertices[i1 + geometryInfo.vertexOffset].normal;
    n[2] = vertices[i2 + geometryInfo.vertexOffset].normal;
    t[0] = vertices[i0 + geometryInfo.vertexOffset].tangent;
    t[1] = vertices[i1 + geometryInfo.vertexOffset].tangent;
    t[2] = vertices[i2 + geometryInfo.vertexOffset].tangent;
    for(int i=0; i<3; ++i) {
        const vec3 wNormal = normalize((o2wn * vec4(n[i], 0)).xyz);
        const vec3 wTangent = normalize((o2w * vec4(t[i], 0)).xyz);
        vec3 wBitangent = cross(wNormal, wTangent) * geometryInfo.oddNegativeScaling;
        TBN[i] = mat3(wTangent, wBitangent, wNormal);
    }
    hit.TBN = barycentrics.x * TBN[0] + barycentrics.y * TBN[1] + barycentrics.z * TBN[2];
    // Return hit geometry
    return hit;
}

#endif