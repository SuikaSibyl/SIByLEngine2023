#ifndef _CLOSEST_HIT_COMMON_
#define _CLOSEST_HIT_COMMON_

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "rtCommon.h"

// This will store two of the barycentric coordinates of the intersection when
// closest-hit shaders are called:
hitAttributeEXT vec2 attributes;

// These shaders can access the vertex and index buffers:
// The scalar layout qualifier here means to align types according to the alignment
// of their scalar components, instead of e.g. padding them to std140 rules.
layout(binding = 2, set = 0, scalar) buffer Vertices {
  vec3 vertices[];
};
layout(binding = 3, set = 0, scalar) buffer Indices { uint16_t indices[]; };
layout(binding = 4, set = 0, scalar) buffer Geometry { uvec2 geometryInfo[]; };

// // The payload:
// layout(location = 0) rayPayloadInEXT PassableInfo pld;

// Gets hit info about the object at the intersection. This uses GLSL variables
// defined in closest hit stages instead of ray queries.
HitInfo getObjectHitInfo() {
  HitInfo result;
  // Get the ID of the triangle
  const int primitiveID = gl_PrimitiveID;
  const int instanceID = gl_InstanceID;
  const uvec2 geometryInfoID = geometryInfo[instanceID];

  // Get the indices of the vertices of the triangle
  const uint i0 = indices[3 * primitiveID + 0 + geometryInfoID.y];
  const uint i1 = indices[3 * primitiveID + 1 + geometryInfoID.y];
  const uint i2 = indices[3 * primitiveID + 2 + geometryInfoID.y];

  // Get the vertices of the triangle
  const vec3 v0 = vertices[i0 + geometryInfoID.x/3];
  const vec3 v1 = vertices[i1 + geometryInfoID.x/3];
  const vec3 v2 = vertices[i2 + geometryInfoID.x/3];

  // Get the barycentric coordinates of the intersection
  vec3 barycentrics = vec3(0.0, attributes.x, attributes.y);
  barycentrics.x    = 1.0 - barycentrics.y - barycentrics.z;

  // Compute the coordinates of the intersection
  result.objectPosition = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
  // Transform from object space to world space:
  result.worldPosition = gl_ObjectToWorldEXT * vec4(result.objectPosition, 1.0f);

  // Compute the normal of the triangle in object space, using the right-hand rule:
  //    v2      .
  //    |\      .
  //    | \     .
  //    |/ \    .
  //    /   \   .
  //   /|    \  .
  //  L v0---v1 .
  // n
  const vec3 objectNormal = cross(v1 - v0, v2 - v0);
  // Transform normals from object space to world space. These use the transpose of the inverse matrix,
  // because they're directions of normals, not positions:
  result.worldNormal = normalize((objectNormal * gl_WorldToObjectEXT).xyz);

  // Flip the normal so it points against the ray direction:
  const vec3 rayDirection = gl_WorldRayDirectionEXT;
  result.worldNormal      = faceforward(result.worldNormal, rayDirection, result.worldNormal);

  return result;
}

#endif