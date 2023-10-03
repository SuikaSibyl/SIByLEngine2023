#ifndef _SRENDERER_GBUFFER_ADDON_HEADER_
#define _SRENDERER_GBUFFER_ADDON_HEADER_

#include "../../include/common/camera.hlsli"
#include "../../include/common/octahedral.hlsli"
#include "../../include/common/packing.hlsli"
#include "../../include/common/shading.hlsli"

#ifndef _GBUFFER_RW_ENABLE_
#define RESOURCE_TYPE Texture2D
#else
#define RESOURCE_TYPE RWTexture2D
#endif

RESOURCE_TYPE<float4> t_GBufferPosition;
RESOURCE_TYPE<uint>  t_GBufferNormals;
RESOURCE_TYPE<uint>  t_GBufferGeoNormals;
RESOURCE_TYPE<uint>  t_GBufferDiffuseAlbedo;
RESOURCE_TYPE<uint>  t_GBufferSpecularRough;
RESOURCE_TYPE<float4> t_MotionVectors;
RESOURCE_TYPE<float16_t4> t_MaterialInfo;

ShadingSurface GetGBufferSurface(
    in_ref(int2) pixelPosition,
    in_ref(CameraData) cameraData,
    Texture2D<float4> positionTexture,
    Texture2D<uint> normalsTexture,
    Texture2D<uint> geoNormalsTexture,
    Texture2D<uint> diffuseAlbedoTexture,
    Texture2D<uint> specularRoughTexture,
    Texture2D<float16_t4> materialInfo
) {
    ShadingSurface surface = EmptyShadingSurface();
    // outside gbuffer
    if (any(pixelPosition >= getViewportSize(cameraData)))
        return surface;
    // fetch view depth
    const float4 position = positionTexture[pixelPosition];
    surface.viewDepth = position.w;
    if (surface.viewDepth == k_background_depth)
        return surface;
    // fetch further gbuffer data
    surface.shadingNormal = Unorm32OctahedronToUnitVector(normalsTexture[pixelPosition]);
    surface.geometryNormal = Unorm32OctahedronToUnitVector(geoNormalsTexture[pixelPosition]);
    surface.diffuseAlbedo = Unpack_R11G11B10_UFLOAT(diffuseAlbedoTexture[pixelPosition]).rgb;
    float4 specularRough = Unpack_R8G8B8A8_Gamma_UFLOAT(specularRoughTexture[pixelPosition]);
    surface.specularF0 = specularRough.rgb;
    surface.roughness = specularRough.a;
    surface.worldPos = position.xyz;
    surface.viewDir = normalize(cameraData.posW.xyz - surface.worldPos);
    surface.transmissionFactor = materialInfo[pixelPosition].w;
    const uint packed_z = asuint16(materialInfo[pixelPosition].z);
    surface.faceForward = (packed_z & 0x8000) != 0;
    surface.bsdfID = packed_z & 0x7FFF;
    return surface;
}

ShadingSurface GetGBufferSurface(
    in_ref(int2) pixelPosition,
    in_ref(CameraData) cameraData,
    RWTexture2D<float4> positionTexture,
    RWTexture2D<uint> normalsTexture,
    RWTexture2D<uint> geoNormalsTexture,
    RWTexture2D<uint> diffuseAlbedoTexture,
    RWTexture2D<uint> specularRoughTexture,
    RWTexture2D<float16_t4> materialInfo)
{
    ShadingSurface surface = EmptyShadingSurface();
    // outside gbuffer
    if (any(pixelPosition >= getViewportSize(cameraData)))
        return surface;
    // fetch view depth
    const float4 position = positionTexture[pixelPosition];
    surface.viewDepth = position.w;
    if (surface.viewDepth == k_background_depth)
        return surface;
    // fetch further gbuffer data
    surface.shadingNormal = Unorm32OctahedronToUnitVector(normalsTexture[pixelPosition]);
    surface.geometryNormal = Unorm32OctahedronToUnitVector(geoNormalsTexture[pixelPosition]);
    surface.diffuseAlbedo = Unpack_R11G11B10_UFLOAT(diffuseAlbedoTexture[pixelPosition]).rgb;
    float4 specularRough = Unpack_R8G8B8A8_Gamma_UFLOAT(specularRoughTexture[pixelPosition]);
    surface.specularF0 = specularRough.rgb;
    surface.roughness = specularRough.a;
    surface.worldPos = position.xyz;
    surface.viewDir = normalize(cameraData.posW.xyz - surface.worldPos);
    surface.transmissionFactor = materialInfo[pixelPosition].w;
    const uint packed_z = asuint(materialInfo[pixelPosition].z);
    surface.faceForward = (packed_z & 0x8000) != 0;
    surface.bsdfID = packed_z & 0x7FFF;
    return surface;
}

float3 GetMotionVector(in_ref(int2) pixelPosition) {
    return t_MotionVectors[pixelPosition].xyz;
}
float GetFWidthDepth(in_ref(int2) pixelPosition) {
    return t_MotionVectors[pixelPosition].w;
}
float3 GetDiffuseAlbedo(in_ref(int2) pixelPosition) {
    return Unpack_R11G11B10_UFLOAT(t_GBufferDiffuseAlbedo[pixelPosition]).rgb;
}

ShadingSurface GetGBufferSurface(
    in_ref(int2) pixelPosition,
    in_ref(CameraData) cameraData)
{
    return GetGBufferSurface(
        pixelPosition,
        cameraData,
        t_GBufferPosition,
        t_GBufferNormals,
        t_GBufferGeoNormals,
        t_GBufferDiffuseAlbedo,
        t_GBufferSpecularRough,
        t_MaterialInfo);
}

// The motion vectors rendered by the G-buffer pass match what is expected by NRD and DLSS.
// In case of dynamic resolution, there is a difference that needs to be corrected...
//
// The rendered motion vectors are computed as:
//     (previousUV - currentUV) * currentViewportSize
//
// The motion vectors necessary for pixel reprojection are:
//     (previousUV * previousViewportSize - currentUV * currentViewportSize)
//
float3 convertMotionVectorToPixelSpace(
    in_ref(CameraData) cameraData,
    in_ref(CameraData) prevCameraData,
    in_ref(int2) pixelPosition,
    in_ref(float3) motionVector)
{
    float2 curerntPixelCenter = float2(pixelPosition.xy) + 0.5;
    float2 previousPosition = curerntPixelCenter + motionVector.xy;
    previousPosition *= getViewportSize(prevCameraData) * getInvViewportSize(cameraData);
    motionVector.xy = previousPosition - curerntPixelCenter;
    return motionVector;
}

float3 GetGeometryNormal(in_ref(int2) pixelPosition) {
    return Unorm32OctahedronToUnitVector(t_GBufferGeoNormals[pixelPosition]);
}
float3 GetNormal(in_ref(int2) pixelPosition) {
    return Unorm32OctahedronToUnitVector(t_GBufferNormals[pixelPosition]);
}
float GetViewDepth(in_ref(int2) pixelPosition) {
    return t_GBufferPosition[pixelPosition].w;
}
float4 GetSpecularRoughness(in_ref(int2) pixelPosition) {
    return Unpack_R8G8B8A8_Gamma_UFLOAT(t_GBufferSpecularRough[pixelPosition]);
}
// // Load a sample from the previous G-buffer.
// ShadingSurface GetGBufferSurface(int2 pixelPosition, bool previousFrame) {
//     ShadingSurface surface = EmptyShadingSurface();
//     // We do not have access to the current G-buffer in this sample because it's using
//     // a single render pass with a fused resampling kernel, so just return an invalid surface.
//     // This should never happen though, as the fused kernel doesn't call RAB_GetGBufferSurface(..., false)
//     if (!previousFrame)
//         return surface;

//     const PlanarViewConstants view = g_Const.prevView;

//     if (any(pixelPosition >= view.viewportSize))
//         return surface;

//     surface.viewDepth = t_PrevGBufferDepth[pixelPosition];

//     if (surface.viewDepth == BACKGROUND_DEPTH)
//         return surface;

//     surface.normal = octToNdirUnorm32(t_PrevGBufferNormals[pixelPosition]);
//     surface.geoNormal = octToNdirUnorm32(t_PrevGBufferGeoNormals[pixelPosition]);
//     surface.diffuseAlbedo = Unpack_R11G11B10_UFLOAT(t_PrevGBufferDiffuseAlbedo[pixelPosition]).rgb;
//     float4 specularRough = Unpack_R8G8B8A8_Gamma_UFLOAT(t_PrevGBufferSpecularRough[pixelPosition]);
//     surface.specularF0 = specularRough.rgb;
//     surface.roughness = specularRough.a;
//     surface.worldPos = viewDepthToWorldPos(view, pixelPosition, surface.viewDepth);
//     surface.viewDir = normalize(g_Const.view.cameraDirectionOrPosition.xyz - surface.worldPos);
//     surface.diffuseProbability = getSurfaceDiffuseProbability(surface);

//     return surface;
// }

#endif // !_SRENDERER_GBUFFER_ADDON_HEADER_