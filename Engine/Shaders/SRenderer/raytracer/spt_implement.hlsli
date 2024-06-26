#ifndef _SRENDERER_SPT_HEALPER_HEADER_
#define _SRENDERER_SPT_HEALPER_HEADER_

#include "../include/scene_descriptor_set.hlsli"
#include "spt_interface.hlsli"

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
    const GeometryInfo geometry = geometries[geometryID];
    
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
    float4 tangentWS = float4(normalize(mul(float4(tangentOS, 0), o2w).xyz), GetOddNegativeScaling(geometry));
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

/**
 * Fetches the geometry hit info for a given geometry ID and primitive ID.
 * Specially, the noraml is adaptive to face the given ray.
 * @param geometryID The geometry ID.
 * @param bary The barycentric coordinates.
 * @param primitiveID The primitive ID.
 * @param ray The ray.
 * @return The geometry hit info.
 */
GeometryHit fetchTrimeshGeometryHit(
    int geometryID,
    in_ref(float3) bary,
    int primitiveID,
    in_ref(Ray) ray
) {
    GeometryHit hit = fetchTrimeshGeometryHit(geometryID, bary, primitiveID);
    const float frontFace = (dot(hit.geometryNormal, ray.direction) < 0) ? 1 : -1;
    SetFaceForward(hit, frontFace == 1);
    hit.shadingNormal *= frontFace;
    hit.geometryNormal *= frontFace;
    return hit;
}

#endif // !_SRENDERER_SPT_HEALPER_HEADER_