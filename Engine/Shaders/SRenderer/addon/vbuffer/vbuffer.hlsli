#ifndef _SRENDERER_ADDON_VBUFFER_HEADER_
#define _SRENDERER_ADDON_VBUFFER_HEADER_

#include "../../include/common/cpp_compatible.hlsli"
#include "../../raytracer/spt_implement.hlsli"

/**
* Visibility Buffer Addon
* ----------------------------------------------
* This addon is used to render the scene into a visibility buffer.
* Theoretically, it should provide enough data to recover the
* scene geometry and material properties.
*
* The visibility buffer is a 2D texture with 128-bit per pixel.
* It will be in RGBA32UInt format, with the following layout:
* 
*/

struct VBufferData {
    float2 barycentric;
    uint geometryID;
    uint primitiveID;
    uint primitiveType;
};

bool IsInvalid(in_ref(VBufferData) data) {
    return data.primitiveID == 0xFFFFFFFF;
}

uint4 PackVBufferData(in_ref(VBufferData) data) {
    uint4 result;
    result.x = data.primitiveID;
    result.y = data.geometryID + (data.primitiveType << 26);
    result.z = asuint(data.barycentric.x);
    result.w = asuint(data.barycentric.y);
    return result;
}

VBufferData UnpackVBufferData(in_ref(in) uint4 data) {
    VBufferData result;
    result.primitiveID = data.x;
    result.geometryID = data.y & 0x3FFFFFF;
    result.primitiveType = data.y >> 26;
    result.barycentric.x = asfloat(data.z);
    result.barycentric.y = asfloat(data.w);
    return result;
}

GeometryHit GetGeometryHit(in_ref(VBufferData) data) {
    return fetchTrimeshGeometryHit(
        int(data.geometryID),
        float3(1 - data.barycentric.x - data.barycentric.y, data.barycentric.xy),
        int(data.primitiveID));
}

GeometryHit GetGeometryHit(
    in_ref(VBufferData) data,
    in_ref(Ray) ray
) {
    return fetchTrimeshGeometryHit(
        int(data.geometryID),
        float3(1 - data.barycentric.x - data.barycentric.y, data.barycentric.xy),
        int(data.primitiveID), ray);
}

float3 GetVBufferPosition(in_ref(VBufferData) data) {
    const float3 bary = float3(1 - data.barycentric.x - data.barycentric.y, data.barycentric.xy);
    const GeometryInfo geometry = geometries[data.geometryID];
    const int3 index = int3(fetchTriangleIndices(geometry, int(data.primitiveID)));
    float3 vertexPositions[3];
    vertexPositions[0] = fetchVertexPosition(index[0] + int(geometry.vertexOffset));
    vertexPositions[1] = fetchVertexPosition(index[1] + int(geometry.vertexOffset));
    vertexPositions[2] = fetchVertexPosition(index[2] + int(geometry.vertexOffset));
    const float4x4 o2w = ObjectToWorld(geometry);
    const float3 positionOS = interpolate(vertexPositions, bary);
    const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;
    return positionWS;
}

#endif // _SRENDERER_ADDON_VBUFFER_HEADER_