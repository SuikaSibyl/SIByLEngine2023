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
const Texture2D<float4> u_vpl_position;
const Texture3D<int> u_inverse_index;
const StructuredBuffer<uint> u_vplCounter;

RWStructuredBuffer<float4> u_RepresentPixel;
RWStructuredBuffer<float4> u_RepresentVPL;
RWStructuredBuffer<uint4> u_IndirectArgs;

#define VPL_BOUND_ENABLE
#ifdef VPL_BOUND_ENABLE
RWStructuredBuffer<uint> u_BoundMin;
RWStructuredBuffer<uint> u_BoundMax;
#endif // VPL_BOUND_ENABLE
// RWStructuredBuffer<uint> u_DirectionHash;

uint PackInt2ToUInt(int2 xy) {
    return (uint(xy.x) << 16) | uint(xy.y);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int3 Gid: SV_GroupID
) {
    const int2 pixelID = DTid.xy;

    // Grab the bsdf sampled buffer
    const float4 vpl_position = u_vpl_position[pixelID];
    if (!all(vpl_position == float4(0))) {
        const int3 vxID = int3(WorldToVoxel(vpl_position.xyz, info) * info.volumeDimension);
        const int compactID = u_inverse_index[vxID];
        // Inject the voxel
        u_RepresentVPL[compactID] = vpl_position;
    }
    
    // Choosing 128 pixels for row sampling
    if (pixelID.x < 16 && pixelID.y < 8) {
        // stratified sampling from 16 x 8 grid
        RandomSamplerState RNG = InitRandomSampler(pixelID, rand_seed);
        const int flatten_id = pixelID.y * 16 + pixelID.x;
        const float2 cell_size = float2(resolution) / float2(16, 8);
        const float2 pixel = cell_size * (float2(pixelID) + GetNextRandomFloat2(RNG));
        const int2 pixel_int = clamp(int2(pixel), int2(0, 0), resolution - int2(1, 1));
        float4 represent_pixel = u_shadingpoint[pixel_int];
        if (any(represent_pixel == k_inf)) represent_pixel = float4(0);
        u_RepresentPixel[flatten_id] = represent_pixel;
    }
    
    // Indirect arguements
    if (all(pixelID == int2(0, 0))) {
        const uint vplCount = u_vplCounter[0];
        u_IndirectArgs[0] = uint4((vplCount + 255) / 256, 1, 1, vplCount);
        u_IndirectArgs[1] = int4((vplCount * 2 - 1 + 255) / 256, 1, 1, vplCount);
        u_IndirectArgs[2] = uint4(4, (vplCount + 7) / 8, 1, vplCount);
    }
}