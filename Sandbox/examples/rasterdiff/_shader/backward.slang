#include "common.hlsli"
#include "common/random.hlsli"
#include "srenderer/scene-binding.hlsli"

// Albedo
StructuredBuffer<uint> GroundTruth;
Texture2D<float4> RenderedPlus;
Texture2D<float4> RenderedMinus;
Texture2D<uint2> IndexPlus;
Texture2D<uint2> IndexMinus;

RWByteAddressBuffer dAlbedo;
RWByteAddressBuffer dPosition;

[[vk::push_constant]]
cbuffer PushConstants {
    int random_seed;
    float epsilon_albedo;
    float epsilon_pos;
    int camera_index;
};

float3 load_groundtruth(int2 pixel, int index) {
    int index = pixel.y * 800 + pixel.x + index * 800 * 800;
    uint pixel_packed = GroundTruth[index];
    return float3(
        (pixel_packed & 0x000000FF) / 255.f,
        ((pixel_packed & 0x0000FF00) >> 8) / 255.f,
        ((pixel_packed & 0x00FF0000) >> 16) / 255.f
    );
}

void add_triangle_derivative(int primitiveID, float differential) {
    const GeometryData geometry = GPUScene_geometry[0];
    const int3 index = int3(fetchTriangleIndices(geometry, primitiveID));
    
    const int vertexID_0 = index[0] + int(geometry.vertexOffset);
    const int vertexID_1 = index[1] + int(geometry.vertexOffset);
    const int vertexID_2 = index[2] + int(geometry.vertexOffset);

    add_vertex_derivative(vertexID_0, differential);
    add_vertex_derivative(vertexID_1, differential);
    add_vertex_derivative(vertexID_2, differential);
}

void add_vertex_derivative(int vertexID, float differential) {
    RandomSamplerState RNG = InitRandomSampler(vertexID, random_seed);
    const float3 pos_perturb = epsilon_pos * float2sign(GetNextRandomFloat3(RNG));
    const float3 pos_derivative = differential / (2 * pos_perturb);
    dPosition.InterlockedAddF32(4 * (vertexID * 3 + 0), pos_derivative.x);
    dPosition.InterlockedAddF32(4 * (vertexID * 3 + 1), pos_derivative.y);
    dPosition.InterlockedAddF32(4 * (vertexID * 3 + 2), pos_derivative.z);
}

void add_texel_derivative(int texelID, float differential) {
    RandomSamplerState RNG = InitRandomSampler(texelID, random_seed);
    const float3 pos_perturb = epsilon_pos * float2sign(GetNextRandomFloat3(RNG));
    const float3 albedo_perturb = epsilon_albedo * float2sign(GetNextRandomFloat3(RNG));
    const float3 albedo_derivative = differential / (2 * albedo_perturb);
    dAlbedo.InterlockedAddF32(4 * (texelID * 3 + 0), albedo_derivative.x);
    dAlbedo.InterlockedAddF32(4 * (texelID * 3 + 1), albedo_derivative.y);
    dAlbedo.InterlockedAddF32(4 * (texelID * 3 + 2), albedo_derivative.z);
}

int2 unpack_texel(uint texel) {
    // pack: v.x | (v.y << 16)
    return int2(texel & 0xFFFF, (texel >> 16) & 0xFFFF);
}

uint texel_index(int2 pixel) {
    return pixel.y * 1024 + pixel.x;
} 

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    const int2 pixel = int2(dtid.x, dtid.y);
    float3 gt = load_groundtruth(pixel, camera_index);
    if (!all(gt == 0)) gt = float3(0.5);
    const float3 rendered_plus = RenderedPlus[pixel].rgb;
    const float3 rendered_minus = RenderedMinus[pixel].rgb;

    const uint2 index_plus = IndexPlus[pixel];
    const uint2 index_minus = IndexMinus[pixel];
    const uint primitvie_plus = index_plus.x;
    const uint primitvie_minus = index_minus.x;
    const uint texel_plus = texel_index(unpack_texel(index_plus.y));
    const uint texel_minus = texel_index(unpack_texel(index_minus.y));
    
    const float l2_plus = dot(gt - rendered_plus, gt - rendered_plus);
    const float l2_minus = dot(gt - rendered_minus, gt - rendered_minus);
    const float differential = l2_plus - l2_minus;

    add_triangle_derivative(primitvie_plus, differential);
    add_triangle_derivative(primitvie_minus, differential);
    add_texel_derivative(texel_plus, differential);
    add_texel_derivative(texel_minus, differential);
}