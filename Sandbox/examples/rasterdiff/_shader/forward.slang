#include "srenderer/scene-binding.hlsli"
#include "common/hashing.hlsli"
#include "common/packing.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    uint geometry_idx;
    uint camera_idx;
};

Texture2D<float4> u_texture;

float4 loadTexture2D(Texture2D<float4> texture, float2 uv, out int2 texel) {
    int2 resolution;
    texture.GetDimensions(resolution.x, resolution.y);
    texel = int2(uv * float2(resolution));
    return u_texture[texel];
}

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

uint pack_uint2(uint2 v) {
    return v.x | (v.y << 16);
}

[shader("fragment")]
void fragmentMain(
    in CoarseVertex coarseVertex: CoarseVertex,
    nointerpolation in int primitiveID: SV_PrimitiveID,
    in bool isFrontFace: SV_IsFrontFace,
    float4 svPos: SV_POSITION,
    in float3 bary: SV_Barycentrics,
    out float4 o_color: SV_Target0,
    out uint2 o_index: SV_Target1) : SV_Target
{
    const float2 pixel = float2(svPos.x, svPos.y) - float2(0.5);
    const Ray ray = generateRay(pixel, resolution, false, GPUScene_camera[camera_idx]);
    GeometryHit hit = fetchTrimeshGeometryHit(coarseVertex.geoID, bary, primitiveID, ray);
    float3 radiance = float3(0, 0, 0);
    int2 texel = int2(0, 0);
    if (HasHit(hit)) {
        radiance = loadTexture2D(u_texture, hit.texcoord, texel).rgb;
        // const uint colorPacked = murmur3(primitiveID);
        // radiance = radiance * 0.0001 + UNPACK_FLOAT3_UINT(colorPacked);
    } else {
        // const uint colorPacked = murmur3(vhash.y);
        // color = UNPACK_FLOAT3_UINT(colorPacked);
    }

    o_color = float4(radiance, 1.0);
    o_index = uint2(primitiveID, pack_uint2(texel));
}
