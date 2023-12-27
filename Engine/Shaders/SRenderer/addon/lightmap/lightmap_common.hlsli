#ifndef SRENDERER_ADDON_LIGHTMAP_COMMON_HLSLI_HEADER_
#define SRENDERER_ADDON_LIGHTMAP_COMMON_HLSLI_HEADER_

#include "../../raytracer/spt_interface.hlsli"

StructuredBuffer<float2> UV2Buffer;

float2 fetchVertexUV2(int vertexIndex) {
    return UV2Buffer[vertexIndex];
}

float2 fetchVertexUV2(GeometryHit hit) {
    const GeometryInfo geometry = geometries[hit.geometryID];
    const int3 index = int3(fetchTriangleIndices(geometry, hit.primitiveID));
    float2 vertexUV2s[3];
    vertexUV2s[0] = fetchVertexTexCoord(index[0] + int(geometry.vertexOffset));
    vertexUV2s[1] = fetchVertexTexCoord(index[1] + int(geometry.vertexOffset));
    vertexUV2s[2] = fetchVertexTexCoord(index[2] + int(geometry.vertexOffset));
    const float3 bary = float3(1.f - hit.barycentric.x - hit.barycentric.y, hit.barycentric);
    return interpolate(vertexUV2s, bary);
}

#endif // !SRENDERER_ADDON_LIGHTMAP_COMMON_HLSLI_HEADER_