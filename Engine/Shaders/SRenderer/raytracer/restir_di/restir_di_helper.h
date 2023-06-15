#ifndef _SRENDERER_RESTIR_DI_HELPER_
#define _SRENDERER_RESTIR_DI_HELPER_

/** Generate a ray to test visibility between a surface and a light sample. */
Ray setupVisibilityRay(in const GBufferSurface surface, in const DILightSample lightSample, float offset) {
    const vec3 L = lightSample.position - surface.worldPos;
    Ray ray; ray.tMin = offset; ray.tMax = length(L) - offset;
    ray.direction = normalize(L); ray.origin = surface.worldPos;
    return ray;
}
Ray setupVisibilityRay(in const GBufferSurface surface, in const DILightSample lightSample) {
    return setupVisibilityRay(surface, lightSample, 0.001); }

/** Tests the visibility between a surface and a light sample.
* @return: Returns true if there is nothing between them. */
bool GetConservativeVisibility(in const GBufferSurface surface, in const DILightSample lightSample) {
    Ray ray = setupVisibilityRay(surface, lightSample);
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(rayQuery,         // Ray query
                    tlas,                   // Top-level acceleration structure
                    gl_RayFlagsOpaqueEXT,   // Ray flags, here saying "treat all geometry as opaque"
                    0xFF,                   // 8-bit instance mask, here saying "trace against all instances"
                    ray.origin,             // Ray origin
                    ray.tMin,               // Minimum t-value
                    ray.direction,          // Ray direction
                    ray.tMax);              // Maximum t-value
    while(rayQueryProceedEXT(rayQuery)) {
        // TODO :: Fix this to use the correct intersection for non-triangle geometry
    }
    bool visible = rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionNoneEXT;
    return visible;
}

/** Reads the G-buffer, either the current one or the previous one, and returns a surface.
* If the provided pixel position is outside of the viewport bounds, the surface
* should indicate that it's invalid when IsSurfaceValid is called on it. */
GBufferSurface GetGBufferSurface(ivec2 pixelPosition, bool previousFrame) {
    if(previousFrame) {
        return GetGBufferSurface(
            pixelPosition,
            gUniform.prevView,
            t_PrevGBufferDepth, 
            t_PrevGBufferNormals, 
            t_PrevGBufferGeoNormals, 
            t_PrevGBufferDiffuseAlbedo, 
            t_PrevGBufferSpecularRough);
    } else {
        return GetGBufferSurface(
            pixelPosition, 
            gUniform.view, 
            t_GBufferDepth, 
            t_GBufferNormals, 
            t_GBufferGeoNormals, 
            t_GBufferDiffuseAlbedo, 
            t_GBufferSpecularRough);
    }
}

#endif // !_SRENDERER_RESTIR_DI_HELPER_