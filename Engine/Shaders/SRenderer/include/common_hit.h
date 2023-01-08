#ifndef _SRENDERER_COMMON_HIT_HEADER_
#define _SRENDERER_COMMON_HIT_HEADER_

#include "common_trace.h"
#include "common_custom_primitives.h"

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
    const mat4 o2w = ObjectToWorld(geometryInfo);
    const mat4 o2wn = ObjectToWorldNormal(geometryInfo);
    // Get matID
    hit.matID = geometryInfo.materialID;
#if (PRIMITIVE_TYPE == PRIMITIVE_SPHERE)
    // ray data
    const vec3 ray_origin    = gl_WorldRayOriginEXT;
    const vec3 ray_direction = gl_WorldRayDirectionEXT;
    // Sphere data
    const vec3  sphere_center = (o2w * vec4(0,0,0,1)).xyz;
    const float sphere_radius = length((o2w * vec4(1,0,0,1)).xyz - sphere_center);
    // Record the intersection
    const vec3 hitPoint = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    const vec3 geometric_normal = normalize(hitPoint - sphere_center);
    const vec3 cartesian = geometric_normal;
    // We use the spherical coordinates as uv
    // We use the convention that y is up axis.
    // https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
    const float elevation = acos(clamp(cartesian.y, -1., 1.));
    const float azimuth = atan2(cartesian.z, cartesian.x);
    hit.worldPosition = geometric_normal;
    hit.uv = vec2(-azimuth * k_inv_2_pi, elevation * k_inv_pi);
    const vec3 wNormal = geometric_normal;
    const vec3 wTangent = cross(geometric_normal, vec3(0,1,0));
    vec3 wBitangent = cross(wNormal, wTangent) * geometryInfo.oddNegativeScaling;
    hit.TBN = mat3(wTangent, wBitangent, wNormal);
#elif (PRIMITIVE_TYPE == PRIMITIVE_TRIANGLE)
    // Get the indices of the vertices of the triangle
    const uint i0 = indices[3 * primitiveID + 0 + geometryInfo.indexOffset];
    const uint i1 = indices[3 * primitiveID + 1 + geometryInfo.indexOffset];
    const uint i2 = indices[3 * primitiveID + 2 + geometryInfo.indexOffset];
    // Get the vertices of the triangle
    const vec3 v0 = vertices[i0 + geometryInfo.vertexOffset].position;
    const vec3 v1 = vertices[i1 + geometryInfo.vertexOffset].position;
    const vec3 v2 = vertices[i2 + geometryInfo.vertexOffset].position;
    // Get the barycentric coordinates of the intersection
    vec3 barycentrics = vec3(0.0, attributes.x, attributes.y);
    barycentrics.x    = 1.0 - barycentrics.y - barycentrics.z;
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
#endif
    // Return hit geometry
    return hit;
}

#endif