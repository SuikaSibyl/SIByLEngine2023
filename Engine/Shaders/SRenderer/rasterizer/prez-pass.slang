#include "../include/scene_descriptor_set.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint geometry_idx;
    } pConst;
};

struct AssembledVertex { int vertexId : SV_VertexId; };
struct VertexStageOutput { float4 sv_position : SV_Position; };

[shader("vertex")]
VertexStageOutput vertexMain_Indirect(AssembledVertex assembledVertex) {
    const GeometryInfo geometry = geometries[pConst.geometry_idx];
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