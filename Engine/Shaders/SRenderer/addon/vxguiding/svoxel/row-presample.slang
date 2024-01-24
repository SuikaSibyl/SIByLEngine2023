#include "../../vbuffer/vbuffer.hlsli"
#include "../../vxgi/include/conetrace_utils.hlsli"
#include "../../vxgi/include/vxgi_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    // int2 map_size;
    int2 resolution;
    int rand_seed;
};

const Texture2D<float4> u_shadingpoint;
const StructuredBuffer<uint> u_vplCounter;

RWStructuredBuffer<float4> u_RepresentPixel;
RWStructuredBuffer<uint4> u_IndirectArgs;

[shader("compute")]
[numthreads(16, 8, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int3 Gid: SV_GroupID
) {
    const int2 pixelID = DTid.xy;

    // stratified sampling from 16 x 8 grid
    RandomSamplerState RNG = InitRandomSampler(pixelID, rand_seed);
    const int flatten_id = pixelID.y * 16 + pixelID.x;
    const float2 cell_size = float2(resolution) / float2(16, 8);
    const float2 pixel = cell_size * (float2(pixelID) + GetNextRandomFloat2(RNG));
    const int2 pixel_int = clamp(int2(pixel), int2(0, 0), resolution - int2(1, 1));
    float4 represent_pixel = u_shadingpoint[pixel_int];
    if (any(represent_pixel == k_inf)) represent_pixel = float4(0);
    u_RepresentPixel[flatten_id] = represent_pixel;

    // Indirect arguements
    if (all(pixelID == int2(0, 0))) {
        const uint vplCount = u_vplCounter[0];
        u_IndirectArgs[0] = uint4((vplCount + 255) / 256, 1, 1, vplCount);
        u_IndirectArgs[1] = int4((vplCount * 2 - 1 + 255) / 256, 1, 1, vplCount);
        u_IndirectArgs[2] = uint4(4, (vplCount + 7) / 8, 1, vplCount);
    }
}