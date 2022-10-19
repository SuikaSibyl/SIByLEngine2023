float countIntersection(in rayQueryEXT rayQuery) {
    float numIntersections = 0.0;
    while(rayQueryProceedEXT(rayQuery)) {
        numIntersections += 1.0;
    }
    return numIntersections;
}

HitInfo getObjectHitInfo(rayQueryEXT rayQuery) {
  HitInfo result;
  // Get the ID of the triangle
  const int primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
  // Get the indices of the vertices of the triangle
  const uint i0 = indices[3 * primitiveID + 0];
  const uint i1 = indices[3 * primitiveID + 1];
  const uint i2 = indices[3 * primitiveID + 2];
  // Get the vertices of the triangle
  const vec3 v0 = vertices[i0];
  const vec3 v1 = vertices[i1];
  const vec3 v2 = vertices[i2];
  // Get the barycentric coordinates of the intersection
  vec3 barycentrics = vec3(0.0, rayQueryGetIntersectionBarycentricsEXT(rayQuery, true));
  barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;
  // Compute the coordinates of the intersection
  const vec3 objectPos = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
  // For the main tutorial, object space is the same as world space:
  result.objectPosition = objectPos;
    // Transform from object space to world space:
  const mat4x3 objectToWorld = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
  result.worldPosition = objectToWorld * vec4(objectPos, 1.0f);
  // Compute the normal of the triangle in object space, using the right-hand rule:
  const vec3 objectNormal = normalize(cross(v1 - v0, v2 - v0));
  // Transform normals from object space to world space. These use the transpose of the inverse matrix,
  // because they're directions of normals, not positions:
  const mat4x3 objectToWorldInverse = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
  result.worldNormal                = normalize((objectNormal * objectToWorldInverse).xyz);
  // Flip the normal so it points against the ray direction:
  const vec3 rayDirection = rayQueryGetWorldRayDirectionEXT(rayQuery);
  result.worldNormal      = faceforward(result.worldNormal, rayDirection, result.worldNormal);
  return result;
}
