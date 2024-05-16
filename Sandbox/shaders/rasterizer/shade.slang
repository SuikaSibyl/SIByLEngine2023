#include "srenderer/scene-binding.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    uint geometry_idx;
    uint camera_idx;
};

struct AssembledVertex { int vertexId : SV_VertexId; };
struct CoarseVertex { uint geoID; };
struct VertexStageOutput {
    CoarseVertex coarseVertex : CoarseVertex;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex) {
    const GeometryData geometry = GPUScene_geometry[geometry_idx];
    const float4x4 o2w = ObjectToWorld(geometry);
    
    const float3 positionOS = fetchVertexPosition(assembledVertex.vertexId);
    const float3 positionWS = mul(float4(positionOS, 1.0), o2w).xyz;
    const float4 positionCS = mul(float4(positionWS, 1.0f), GPUScene_camera[camera_idx].viewProjMat);
    
    VertexStageOutput output;
    output.coarseVertex.geoID = geometry_idx;
    output.sv_position = positionCS;
    return output;
}

[shader("fragment")]
float4 fragmentMain(
    in CoarseVertex coarseVertex: CoarseVertex,
    nointerpolation in int primitiveID: SV_PrimitiveID,
    in bool isFrontFace: SV_IsFrontFace,
    float4 svPos: SV_POSITION,
    in float3 bary: SV_Barycentrics) : SV_Target
{
    const float2 pixel = float2(svPos.x, svPos.y) - float2(0.5);
    const Ray ray = generateRay(pixel, resolution, false, GPUScene_camera[camera_idx]);
    return float4(ray.direction*0.00001 + float3(1,0,1), 1.0);
}
