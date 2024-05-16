#ifndef _SRENDERER_SCENE_BINDING_
#define _SRENDERER_SCENE_BINDING_

#include "../common/camera.hlsli"
#include "../common/raycast.hlsli"
#include "spt.hlsli"

// geometry info
struct GeometryData {
    uint vertexOffset;
    uint indexOffset;
    uint materialID;
    uint indexSize;
    float surfaceArea;
    uint lightID;
    uint primitiveType;
    float oddNegativeScaling;
    float4 transform[3];
    float4 transformInverse[3];
};

ByteAddressBuffer GPUScene_position;
ByteAddressBuffer GPUScene_vertex;
RWStructuredBuffer<uint32_t> GPUScene_index;
RWStructuredBuffer<GeometryData> GPUScene_geometry;
RWStructuredBuffer<CameraData> GPUScene_camera;
RaytracingAccelerationStructure GPUScene_tlas;

float3 fetchVertexPosition(int vertexIndex) { return GPUScene_position.Load<float3>(vertexIndex * 12); }
float3 fetchVertexNormal(int vertexIndex) { return GPUScene_vertex.Load<float3>(vertexIndex * 32 + 0); }
float3 fetchVertexTangent(int vertexIndex) { return GPUScene_vertex.Load<float3>(vertexIndex * 32 + 12); }
float2 fetchVertexTexCoord(int vertexIndex) { return GPUScene_vertex.Load<float2>(vertexIndex * 32 + 24); }
uint3 fetchTriangleIndices(in const GeometryData geoInfo, int triangleIndex) { return uint3(
    GPUScene_index[geoInfo.indexOffset + triangleIndex * 3 + 0],
    GPUScene_index[geoInfo.indexOffset + triangleIndex * 3 + 1],
    GPUScene_index[geoInfo.indexOffset + triangleIndex * 3 + 2]);
}

float4x4 ObjectToWorld(in const GeometryData geometry) {
    return transpose(float4x4(geometry.transform[0], geometry.transform[1], geometry.transform[2], float4(0, 0, 0, 1)));
}

float4x4 WorldToObject(in const GeometryData geometry) {
    return transpose(float4x4(geometry.transformInverse[0], geometry.transformInverse[1], geometry.transformInverse[2], float4(0, 0, 0, 1)));
}

float4x4 ObjectToWorldNormal(in const GeometryData geometry) {
    return float4x4(geometry.transformInverse[0], geometry.transformInverse[1], geometry.transformInverse[2], float4(0, 0, 0, 1));
}

/**
 * Fetches the geometry hit info for a given geometry ID and primitive ID.
 * @param geometryID The geometry ID.
 * @param bary The barycentric coordinates.
 * @param primitiveID The primitive ID.
 * @return The geometry hit info.
 */
GeometryHit fetchTrimeshGeometryHit(
    int geometryID,
    in_ref(float3) bary,
    int primitiveID
) {
    const GeometryData geometry = GPUScene_geometry[geometryID];
    
    GeometryHit hit;
    hit.barycentric = bary.yz;
    hit.primitiveID = primitiveID;
    hit.geometryID = geometryID;

    const int3 index = int3(fetchTriangleIndices(geometry, primitiveID));

    float3 vertexPositions[3];
    vertexPositions[0] = fetchVertexPosition(index[0] + int(geometry.vertexOffset));
    vertexPositions[1] = fetchVertexPosition(index[1] + int(geometry.vertexOffset));
    vertexPositions[2] = fetchVertexPosition(index[2] + int(geometry.vertexOffset));
    
    const float4x4 o2w = ObjectToWorld(geometry);
    const float3 positionOS = interpolate(vertexPositions, bary);
    const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;
    hit.position = positionWS;

    float2 vertexUVs[3];
    vertexUVs[0] = fetchVertexTexCoord(index[0] + int(geometry.vertexOffset));
    vertexUVs[1] = fetchVertexTexCoord(index[1] + int(geometry.vertexOffset));
    vertexUVs[2] = fetchVertexTexCoord(index[2] + int(geometry.vertexOffset));
    vertexUVs[0] = frac(vertexUVs[0]);
    vertexUVs[1] = frac(vertexUVs[1]);
    vertexUVs[2] = frac(vertexUVs[2]);
    float2 uv = interpolate(vertexUVs, bary);
    hit.texcoord = uv;
    
    float3 objectSpaceFlatNormal = normalize(cross(
        vertexPositions[1] - vertexPositions[0],
        vertexPositions[2] - vertexPositions[0]));

    float4x4 o2wn = ObjectToWorldNormal(geometry);
    const float3 flatNormal = normalize(mul(float4(objectSpaceFlatNormal, 0.0), o2wn).xyz);
    hit.geometryNormal = flatNormal;

    float3 normals[3];
    normals[0] = fetchVertexNormal(index[0] + int(geometry.vertexOffset));
    normals[1] = fetchVertexNormal(index[1] + int(geometry.vertexOffset));
    normals[2] = fetchVertexNormal(index[2] + int(geometry.vertexOffset));
    float3 vertexNormalOS = interpolate(normals, bary);
    float3 gvertexNormalWS = normalize(mul(float4(vertexNormalOS, 0.0), o2wn).xyz);
    hit.shadingNormal = gvertexNormalWS;

    float3 tangents[3];
    tangents[0] = fetchVertexTangent(index[0] + int(geometry.vertexOffset));
    tangents[1] = fetchVertexTangent(index[1] + int(geometry.vertexOffset));
    tangents[2] = fetchVertexTangent(index[2] + int(geometry.vertexOffset));
    float3 tangentOS = interpolate(tangents, bary);
    float4 tangentWS = float4(normalize(mul(float4(tangentOS, 0), o2w).xyz), geometry.oddNegativeScaling);
    hit.tangent = tangentWS;

    // compute lambda for ray cone based lod sampling
    const float2 uv_10 = vertexUVs[1] - vertexUVs[0];
    const float2 uv_20 = vertexUVs[2] - vertexUVs[0];
    const float t_a = abs(uv_10.x * uv_20.y - uv_20.x * uv_10.y);
    const float3 edge_10 = mul(float4(vertexPositions[1] - vertexPositions[0], 0.0), o2w).xyz;
    const float3 edge_20 = mul(float4(vertexPositions[2] - vertexPositions[0], 0.0), o2w).xyz;
    const float p_a = length(cross(edge_10, edge_20));
    hit.lambda = 0.5f * log2(t_a / p_a);

    SetHit(hit, true);

    return hit;
}

GeometryHit fetchTrimeshGeometryHit(
    in const int geometryID,
    in const float3 bary,
    in const int primitiveID,
    in const Ray ray
) {
    GeometryHit hit = fetchTrimeshGeometryHit(geometryID, bary, primitiveID);
    const float frontFace = (dot(hit.geometryNormal, ray.direction) < 0) ? 1 : -1;
    SetFaceForward(hit, frontFace == 1);
    hit.shadingNormal *= frontFace;
    hit.geometryNormal *= frontFace;
    return hit;
}

#endif // _SRENDERER_SCENE_BINDING_