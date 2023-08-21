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
    hit.flatNormal = flatNormal;
    
    float3 normals[3];
    normals[0] = fetchVertexNormal(index[0] + int(geometry.vertexOffset));
    normals[1] = fetchVertexNormal(index[1] + int(geometry.vertexOffset));
    normals[2] = fetchVertexNormal(index[2] + int(geometry.vertexOffset));
    float3 geometryNormalOS = interpolate(normals, bary);
    float3 geometryNormalWS = normalize(mul(float4(geometryNormalOS, 0.0), o2wn).xyz);
    hit.geometryNormal = geometryNormalWS;
    
    float3 tangents[3];
    tangents[0] = fetchVertexTangent(index[0] + int(geometry.vertexOffset));
    tangents[1] = fetchVertexTangent(index[1] + int(geometry.vertexOffset));
    tangents[2] = fetchVertexTangent(index[2] + int(geometry.vertexOffset));
    float3 tangentOS = interpolate(tangents, bary);
    float4 tangentWS = float4(normalize(mul(float4(tangentOS, 0), o2w).xyz), geometry.oddNegativeScaling);
    hit.tangent = tangentWS;
        
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
    const float frontFace = (dot(hit.flatNormal, ray.direction) < 0) ? 1 : -1;
    hit.flatNormal *= frontFace;
    hit.geometryNormal *= frontFace;
    return hit;
}

#endif // !_SRENDERER_SPT_HEALPER_HEADER_