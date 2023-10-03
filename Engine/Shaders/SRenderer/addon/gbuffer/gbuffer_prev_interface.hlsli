#ifndef _SRENDERER_GBUFFER_ADDON_PREV_INTERFACE_HEADER_
#define _SRENDERER_GBUFFER_ADDON_PREV_INTERFACE_HEADER_

#include "../../include/common/camera.hlsli"
#include "../../include/common/octahedral.hlsli"
#include "../../include/common/packing.hlsli"
#include "../../include/common/shading.hlsli"
#include "gbuffer_interface.hlsli"

Texture2D<float4> t_PrevGBufferPosition;
Texture2D<uint>  t_PrevGBufferNormals;
Texture2D<uint>  t_PrevGBufferGeoNormals;
Texture2D<uint>  t_PrevGBufferDiffuseAlbedo;
Texture2D<uint> t_PrevGBufferSpecularRough;
Texture2D<float16_t4>  t_PrevGBufferMaterialInfo;

ShadingSurface GetPrevGBufferSurface(
    in_ref(int2) pixelPosition,
    in_ref(CameraData) cameraData)
{
    return GetGBufferSurface(
        pixelPosition,
        cameraData,
        t_PrevGBufferPosition,
        t_PrevGBufferNormals,
        t_PrevGBufferGeoNormals,
        t_PrevGBufferDiffuseAlbedo,
        t_PrevGBufferSpecularRough,
        t_PrevGBufferMaterialInfo);
}

float3 GetGeometryNormalPrev(in_ref(int2) pixelPosition) {
    return Unorm32OctahedronToUnitVector(t_PrevGBufferGeoNormals[pixelPosition]);
}
float3 GetNormalPrev(in_ref(int2) pixelPosition) {
    return Unorm32OctahedronToUnitVector(t_PrevGBufferNormals[pixelPosition]);
}
float GetViewDepthPrev(in_ref(int2) pixelPosition) {
    return t_PrevGBufferPosition[pixelPosition].w;
}

/**
 * Compares two values and returns true if their relative difference is lower than the threshold.
 * Zero or negative threshold makes test always succeed, not fail.
 * @param reference The reference value.
 * @param candidate The candidate value.
 * @param threshold The threshold.
 */
bool CompareRelativeDifference(float reference, float candidate, float threshold) {
    return (threshold <= 0) || abs(reference - candidate) <= threshold * max(reference, candidate);
}

/**
 * See if we will reuse this neighbor or history sample using
 * edge-stopping functions (e.g., per a bilateral filter).
 * @param ourNorm Our surface normal.
 * @param theirNorm The neighbor surface normal.
 * @param ourDepth Our surface depth.
 */
bool IsValidNeighbor(
    in_ref(float3) ourNorm,
    in_ref(float3) theirNorm,
    float ourDepth, float theirDepth,
    float normalThreshold, float depthThreshold
) {
    return (dot(theirNorm.xyz, ourNorm.xyz) >= normalThreshold)
        && CompareRelativeDifference(ourDepth, theirDepth, depthThreshold);
}

// Compares the materials of two surfaces, returns true if the surfaces
// are similar enough that we can share the light reservoirs between them.
// If unsure, just return true.
bool AreMaterialsSimilar(in_ref(ShadingSurface) a, in_ref(ShadingSurface) b) {
    const float roughnessThreshold = 0.5;
    const float reflectivityThreshold = 0.25;
    const float albedoThreshold = 0.25;
    if (!CompareRelativeDifference(a.roughness, b.roughness, roughnessThreshold))
        return false;
    if (abs(luminance(a.specularF0) - luminance(b.specularF0)) > reflectivityThreshold)
        return false;
    if (abs(luminance(a.diffuseAlbedo) - luminance(b.diffuseAlbedo)) > albedoThreshold)
        return false;
    return true;
}

#endif // !_SRENDERER_GBUFFER_ADDON_PREV_INTERFACE_HEADER_