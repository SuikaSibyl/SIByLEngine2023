#version 460
#extension GL_GOOGLE_include_directive : enable
#include "ssrgt_common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload pld;

// This will store two of the barycentric coordinates of the intersection when
// closest-hit shaders are called:
hitAttributeEXT vec2 attributes;

void main() {
    const int primitiveID = gl_PrimitiveID;
    // Get the indices of the vertices of the triangle
    const uint i0 = indices[3 * primitiveID + 0];
    const uint i1 = indices[3 * primitiveID + 1];
    const uint i2 = indices[3 * primitiveID + 2];
    // Get the vertices of the triangle
    const vec3 v0 = vertices[i0];
    const vec3 v1 = vertices[i1];
    const vec3 v2 = vertices[i2];
    // Get the barycentric coordinates of the intersection
    vec3 barycentrics = vec3(0.0, attributes.x, attributes.y);
    barycentrics.x    = 1.0 - barycentrics.y - barycentrics.z;
    // Get position
    vec3 position = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
    // Fill the payload
    pld.position = position.xyz;
    pld.hit = true;
    pld.triangleIndex = primitiveID;
}