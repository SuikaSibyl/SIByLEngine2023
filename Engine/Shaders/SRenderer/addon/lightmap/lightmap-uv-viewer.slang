#include "../../include/common/geometry.hlsli"
#include "../../include/common/math.hlsli"
#include "../../include/common/packing.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/raycast.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "lightmap_common.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint geometry_idx;
    } pConst;
};

struct AssembledVertex { int vertexId : SV_VertexId; };
struct CoarseVertex {
    uint geoID;
};
struct VertexStageOutput {
    CoarseVertex coarseVertex : CoarseVertex;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    const GeometryInfo geometry = geometries[pConst.geometry_idx];
    const float2 positionUV = fetchVertexUV2(assembledVertex.vertexId);

    VertexStageOutput output;
    output.coarseVertex.geoID = pConst.geometry_idx;
    output.sv_position = float4(positionUV * 2 - float2(1), 0.0, 1.0);
    return output;
}

[shader("fragment")]
void fragmentMain(
    in CoarseVertex coarseVertex: CoarseVertex,
    nointerpolation in int primitiveID: SV_PrimitiveID,
    in bool isFrontFace: SV_IsFrontFace,
    in float3 bary: SV_Barycentrics,
    in float4 svPosition: SV_Position,
    // output
    out float4 o_output: SV_Target0)
{
    const GeometryInfo geometry = geometries[coarseVertex.geoID];
    const int3 index = int3(fetchTriangleIndices(geometry, primitiveID));
    const MaterialInfo material = materials[geometry.materialID];

    float2 vertexUVs[3];
    vertexUVs[0] = fetchVertexTexCoord(index[0] + int(geometry.vertexOffset));
    vertexUVs[1] = fetchVertexTexCoord(index[1] + int(geometry.vertexOffset));
    vertexUVs[2] = fetchVertexTexCoord(index[2] + int(geometry.vertexOffset));
    float2 uv = interpolate(vertexUVs, bary);

    // Grab material infomation
    float3 diffuseAlbedo = textures[material.baseOrDiffuseTextureIndex].Sample(uv).rgb;
    
    o_output = float4(diffuseAlbedo, 1);
}