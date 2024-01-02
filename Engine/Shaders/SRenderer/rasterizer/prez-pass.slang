#include "../include/common/indirect_draw.hlsli"
#include "../include/scene_descriptor_set.hlsli"

StructuredBuffer<DrawIndexedIndirectEX> indirect_draws;

struct AssembledVertex { int vertexId : SV_VertexId; };
struct VertexStageOutput { float4 sv_position : SV_Position; };

[shader("vertex")]
VertexStageOutput vertexMain_Indirect(AssembledVertex assembledVertex) {
    const uint draw_idx = 0;
    const uint geometry_idx = indirect_draws[draw_idx].geometryID;

    const GeometryInfo geometry = geometries[geometry_idx];
    const float4x4 o2w = ObjectToWorld(geometry);

    const float3 positionOS = fetchVertexPosition(assembledVertex.vertexId);
    const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;
    const float4 positionCS = mul(float4(positionWS, 1.0f), globalUniform.cameraData.viewProjMat);

    VertexStageOutput output;
    output.sv_position = positionCS;
    return output;
}

[shader("fragment")]
void fragmentMain() {}