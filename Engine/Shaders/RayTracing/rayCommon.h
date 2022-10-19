// Common file defining all the functions used for materials.
// In the next chapter, this file will be shared across closest-hit shaders.
#ifndef VK_MINI_PATH_TRACER_SHADER_COMMON_H
#define VK_MINI_PATH_TRACER_SHADER_COMMON_H

// Info retrieved from a rayQueryEXT by getObjectHitInfo.
struct HitInfo {
    vec3 objectPosition;  // The intersection position in object-space.
    vec3 worldPosition;   // The intersection position in world-space.
    vec3 worldNormal;     // The double-sided triangle normal in world-space.
};

// offsetPositionAlongNormal shifts a point on a triangle surface so that a
// ray bouncing off the surface with tMin = 0.0 is no longer treated as
// intersecting the surface it originated from.
//
// Here's the old implementation of it we used in earlier chapters:
// vec3 offsetPositionAlongNormal(vec3 worldPosition, vec3 normal)
// {
//   return worldPosition + 0.0001 * normal;
// }
//
// However, this code uses an improved technique by Carsten Wächter and
// Nikolaus Binder from "A Fast and Robust Method for Avoiding
// Self-Intersection" from Ray Tracing Gems (verion 1.7, 2020).
// The normal can be negated if one wants the ray to pass through
// the surface instead.

vec3 offsetPositionAlongNormal(vec3 worldPosition, vec3 normal) {
    // Convert the normal to an integer offset.
    const float int_scale = 256.0f;
    const ivec3 of_i      = ivec3(int_scale * normal);
    // Offset each component of worldPosition using its binary representation.
    // Handle the sign bits correctly.
    const vec3 p_i = vec3(  //
        intBitsToFloat(floatBitsToInt(worldPosition.x) + ((worldPosition.x < 0) ? -of_i.x : of_i.x)),
        intBitsToFloat(floatBitsToInt(worldPosition.y) + ((worldPosition.y < 0) ? -of_i.y : of_i.y)),
        intBitsToFloat(floatBitsToInt(worldPosition.z) + ((worldPosition.z < 0) ? -of_i.z : of_i.z)));
    // Use a floating-point offset instead for points near (0,0,0), the origin.
    const float origin     = 1.0f / 32.0f;
    const float floatScale = 1.0f / 65536.0f;
    return vec3(  //
        abs(worldPosition.x) < origin ? worldPosition.x + floatScale * normal.x : p_i.x,
        abs(worldPosition.y) < origin ? worldPosition.y + floatScale * normal.y : p_i.y,
        abs(worldPosition.z) < origin ? worldPosition.z + floatScale * normal.z : p_i.z);
}

// Returns a random diffuse (Lambertian) reflection for a surface with the
// given normal, using the given random number generator state. This is
// cosine-weighted, so directions closer to the normal are more likely to
// be chosen.
vec3 diffuseReflection(vec3 normal, inout uint rngState) {
  const vec3  direction = normal + randomPointInSphere(rngState);
  // Then normalize the ray direction:
  return normalize(direction);
}


#endif  // #ifndef VK_MINI_PATH_TRACER_SHADER_COMMON_H