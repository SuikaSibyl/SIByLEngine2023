import camera;

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        float2 center;
        float2 diag;
    } pConst;
};

[[vk::binding(0, 0)]]
cbuffer GlobalUniforms {
    CameraData gCamera;
}

[[vk::binding(1, 0)]]
Sampler2D<float4> in_color;

struct AssembledVertex  { uint vertexId   : SV_VertexId; };
struct CoarseVertex     { float2 uv; };
struct VertexStageOutput {
    CoarseVertex    coarseVertex    : CoarseVertex;
    float4          sv_position     : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    float2 uvs[] = {
        // triangle 0
        float2(0, 0),
        float2(1, 0),
        float2(0, 1),
        float2(1, 0),
        float2(0, 1),
        float2(1, 1)
    };

    const float2 uv = uvs[assembledVertex.vertexId];
    const float3 vertex = float3(pConst.center + ((uv * 2 - float2(1)) * pConst.diag), -1);
    
    VertexStageOutput output;
    output.coarseVertex.uv = uvs[assembledVertex.vertexId];
    output.sv_position = mul(transpose(gCamera.viewProjMat), float4(vertex, 1));
    return output;
}

[shader("fragment")]
float4 fragmentMain(
    CoarseVertex coarseVertex : CoarseVertex) : SV_Target
{
    const float3 color = in_color.SampleLevel(coarseVertex.uv, 0).xyz;
    return float4(color, 1);
}
