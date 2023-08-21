#ifndef _DDGI_ADDON_DESCRIOPTOR_HEADER_
#define _DDGI_ADDON_DESCRIOPTOR_HEADER_

#include "DDGIVolumeDescGPU.hlsli"

StructuredBuffer<DDGIVolumeDescGPUPacked>   DDGIVolumes;
StructuredBuffer<DDGIVolumeResourceIndices> DDGIVolumeBindless;

// Bindless Resources ---------------------------------------------------------------------------------------

RWTexture2D<float4> RWTex2D[];
RWTexture2DArray<float4> RWTex2DArray[];
RaytracingAccelerationStructure TLAS[];
Texture2D Tex2D[];
Texture2DArray Tex2DArray[];
ByteAddressBuffer ByteAddrBuffer[];

StructuredBuffer<DDGIVolumeDescGPUPacked> GetDDGIVolumeConstants() { return DDGIVolumes; }
StructuredBuffer<DDGIVolumeResourceIndices> GetDDGIVolumeResourceIndices() { return DDGIVolumeBindless; }

Texture2DArray<float4> GetTex2DArray(uint index) { return Tex2DArray[index]; }

#endif // !_DDGI_ADDON_DESCRIOPTOR_HEADER_