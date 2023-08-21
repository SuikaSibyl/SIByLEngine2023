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

RESOURCE_TYPE<float> t_GBufferDepth;
RESOURCE_TYPE<uint>  t_GBufferNormals;
RESOURCE_TYPE<uint>  t_GBufferGeoNormals;
RESOURCE_TYPE<uint>  t_GBufferDiffuseAlbedo;
RESOURCE_TYPE<uint>  t_GBufferSpecularRough;
RESOURCE_TYPE<float4> t_MotionVectors;

float3 viewDepthToWorldPos(
    in_ref(CameraData) cameraData,
    int2 pixelPosition,
    float viewDepth)
{
    float2 uv = (float2(pixelPosition) + 0.5) * getInvViewportSize(cameraData);
    float4 clipPos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1);
    float4 viewPos = mul(clipPos, cameraData.invProjMat);
    viewPos.xy /= viewPos.z;
    viewPos.zw = 1.0;
    viewPos.xyz *= viewDepth;
    return mul(viewPos, cameraData.invViewMat).xyz;
}

ShadingSurface GetGBufferSurface(
    in_ref(int2) pixelPosition,
    in_ref(CameraData) cameraData,
    Texture2D<float> depthTexture,
    Texture2D<uint> normalsTexture,
    Texture2D<uint> geoNormalsTexture,
    Texture2D<uint> diffuseAlbedoTexture,
    Texture2D<uint> specularRoughTexture)
{
    ShadingSurface surface = EmptyShadingSurface();
    // outside gbuffer
    if (any(pixelPosition >= getViewportSize(cameraData)))
        return surface;
    // fetch view depth
    surface.viewDepth = depthTexture[pixelPosition];
    if (surface.viewDepth == k_background_depth)
        return surface;
    // fetch further gbuffer data
    surface.normal = Unorm32OctahedronToUnitVector(normalsTexture[pixelPosition]);
    surface.geoNormal = Unorm32OctahedronToUnitVector(geoNormalsTexture[pixelPosition]);
    surface.diffuseAlbedo = Unpack_R11G11B10_UFLOAT(diffuseAlbedoTexture[pixelPosition]).rgb;
    float4 specularRough = Unpack_R8G8B8A8_Gamma_UFLOAT(specularRoughTexture[pixelPosition]);
    surface.specularF0 = specularRough.rgb;
    surface.roughness = specularRough.a;
    surface.worldPos = viewDepthToWorldPos(cameraData, pixelPosition, surface.viewDepth);
    surface.viewDir = normalize(cameraData.posW.xyz - surface.worldPos);
    surface.diffuseProbability = getSurfaceDiffuseProbability(surface);

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
        t_GBufferDepth,
        t_GBufferNormals,
        t_GBufferGeoNormals,
        t_GBufferDiffuseAlbedo,
        t_GBufferSpecularRough);
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
    return Unorm32OctahedronToUnitVector(t_GBufferNormals[pixelPosition]);
}
float GetViewDepth(in_ref(int2) pixelPosition) {
    return t_GBufferDepth[pixelPosition];
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