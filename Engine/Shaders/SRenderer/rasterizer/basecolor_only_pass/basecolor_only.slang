#include "../../include/common/geometry.hlsli"
#include "../../include/scene_descriptor_set.hlsli"


[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint geometry_idx;
    } pConst;
};

[[vk::binding(0, 0)]]
cbuffer GlobalUniforms {
    CameraData gCamera;
}

[[vk::binding(1, 0)]]
Sampler2D<float4> in_color;

struct AssembledVertex { int vertexId : SV_VertexId; };
struct CoarseVertex {
    float2 uv;
    uint matID;
    float3 normalWS;
    float4 tangentWS;
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
    const float4x4 o2w = ObjectToWorld(geometry);

    const float3 positionOS = fetchVertexPosition(assembledVertex.vertexId);
    const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;
    const float4 positionCS = mul(float4(positionWS, 1.0f), globalUniform.cameraData.viewProjMat);

    VertexStageOutput output;
    output.coarseVertex.uv = fetchVertexTexCoord(assembledVertex.vertexId); ;
    output.sv_position = positionCS;
    return output;
}

[shader("fragment")]
float4 fragmentMain(
    CoarseVertex coarseVertex: CoarseVertex,
    bool isFrontFace: SV_IsFrontFace) : SV_Target
{
    float3x3 TBN = buildTangentToWorld(coarseVertex.tangentWS, coarseVertex.normalWS);
    if (!isFrontFace) TBN[2] = -TBN[2];

    // PrincipledMaterialData material = materials[coarseVertex.matID];

    // vec3 base_color = texture(textures[material.basecolor_opacity_tex], uv).rgb;
    // // vec3 normal = texture(textures[material.normal_bump_tex], uv).rgb;
    // // normal = normalize(normal * 2.0 - 1.0);
    // // normal = normalize(TBN * normal);

    return float4(coarseVertex.uv, 1.0, 1.0);
}
