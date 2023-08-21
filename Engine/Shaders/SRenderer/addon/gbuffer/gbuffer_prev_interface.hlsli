#ifndef _SRENDERER_GBUFFER_ADDON_PREV_INTERFACE_HEADER_
#define _SRENDERER_GBUFFER_ADDON_PREV_INTERFACE_HEADER_

#include "../../include/common/camera.hlsli"
#include "../../include/common/octahedral.hlsli"
#include "../../include/common/packing.hlsli"
#include "../../include/common/shading.hlsli"
#include "gbuffer_interface.hlsli"

Texture2D<float> t_PrevGBufferDepth;
Texture2D<uint>  t_PrevGBufferNormals;
Texture2D<uint>  t_PrevGBufferGeoNormals;
Texture2D<uint>  t_PrevGBufferDiffuseAlbedo;
Texture2D<uint>  t_PrevGBufferSpecularRough;

ShadingSurface GetPrevGBufferSurface(
    in_ref(int2) pixelPosition,
    in_ref(CameraData) cameraData)
{
    return GetGBufferSurface(
        pixelPosition,
        cameraData,
        t_PrevGBufferDepth,
        t_PrevGBufferNormals,
        t_PrevGBufferGeoNormals,
        t_PrevGBufferDiffuseAlbedo,
        t_PrevGBufferSpecularRough);
}

float3 GetGeometryNormalPrev(in_ref(int2) pixelPosition) {
    return Unorm32OctahedronToUnitVector(t_PrevGBufferGeoNormals[pixelPosition]);
}
float GetViewDepthPrev(in_ref(int2) pixelPosition) {
    return t_PrevGBufferDepth[pixelPosition];
}

#endif // !_SRENDERER_GBUFFER_ADDON_PREV_INTERFACE_HEADER_