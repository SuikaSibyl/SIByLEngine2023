float countIntersection(in rayQueryEXT rayQuery) {
    float numIntersections = 0.0;
    while(rayQueryProceedEXT(rayQuery)) {
        numIntersections += 1.0;
    }
    return numIntersections;
}

struct HitInfo {
  vec3 color;
  vec3 worldPosition;
  vec3 worldNormal;
};

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
  result.worldPosition = objectPos;
  // Compute the normal of the triangle in object space, using the right-hand rule:
  const vec3 objectNormal = normalize(cross(v1 - v0, v2 - v0));
  // For the main tutorial, object space is the same as world space:
  result.worldNormal = objectNormal;
  result.color = vec3(0.8f);

  const float dotX = dot(result.worldNormal, vec3(1.0, 0.0, 0.0));
  if(dotX > 0.99) {
    result.color = vec3(0.8, 0.0, 0.0);
  }
  else if(dotX < -0.99) {
    result.color = vec3(0.0, 0.8, 0.0);
  }
  return result;
}
