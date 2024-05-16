#include "subgroup-op.hlsli"

RWStructuredBuffer<float> pmf_texture;
RWStructuredBuffer<float> sampler_buffer;

groupshared float shared_buffer[32];

[shader("compute")]
[numthreads(1024, 1, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gctid: SV_GroupID,
    int gidx: SV_GroupIndex
) {
    const int2 offset = gctid.xy * 32;
    const int2 pixel = offset + int2(gidx % 32, gidx / 32);
    const int index = pixel.y * (33 * 32) + pixel.x;
    const float val = pmf_texture[index * 14 + 0];

    const int superpixel_index = gctid.x + gctid.y * 33;
    const int superpixel_offset = 2048 * superpixel_index;

    const int sgInvocationID = gidx % 32;
    const int sgSubgroupID = gidx / 32;

    float weight = val;
    sampler_buffer[superpixel_offset + 1024 + gidx] = weight;
    weight = __inline_subgroupClusteredAdd2(weight); // 32 -> 16
    weight = __inline_subgroupShuffle(weight, sgInvocationID << 1);
    if (sgInvocationID < 16) sampler_buffer[superpixel_offset + 512 + sgSubgroupID * 16 + sgInvocationID] = weight;
    weight = __inline_subgroupClusteredAdd2(weight); // 16 -> 8
    weight = __inline_subgroupShuffle(weight, sgInvocationID << 1);
    if (sgInvocationID < 8) sampler_buffer[superpixel_offset + 256 + sgSubgroupID * 8 + sgInvocationID] = weight;
    weight = __inline_subgroupClusteredAdd2(weight); // 8 -> 4
    weight = __inline_subgroupShuffle(weight, sgInvocationID << 1);
    if (sgInvocationID < 4) sampler_buffer[superpixel_offset + 128 + sgSubgroupID * 4 + sgInvocationID] = weight;
    weight = __inline_subgroupClusteredAdd2(weight); // 4 -> 2
    weight = __inline_subgroupShuffle(weight, sgInvocationID << 1);
    if (sgInvocationID < 2) sampler_buffer[superpixel_offset + 64 + sgSubgroupID * 2 + sgInvocationID] = weight;
    weight = __inline_subgroupClusteredAdd2(weight); // 2 -> 1
    weight = __inline_subgroupShuffle(weight, sgInvocationID << 1);
    if (sgInvocationID < 1) sampler_buffer[superpixel_offset + 32 + sgSubgroupID * 1 + sgInvocationID] = weight;

    if (sgInvocationID == 0) {
        shared_buffer[sgSubgroupID] = weight;
    }
    
    GroupMemoryBarrierWithGroupSync();

    if (gidx < 32) {
        weight = shared_buffer[gidx];
        weight = __inline_subgroupClusteredAdd2(weight); // 32 -> 16
        weight = __inline_subgroupShuffle(weight, sgInvocationID << 1);
        if (sgInvocationID < 16) sampler_buffer[superpixel_offset + 16 + sgInvocationID] = weight;
        weight = __inline_subgroupClusteredAdd2(weight); // 16 -> 8
        weight = __inline_subgroupShuffle(weight, sgInvocationID << 1);
        if (sgInvocationID < 8) sampler_buffer[superpixel_offset + 8 + sgInvocationID] = weight;
        weight = __inline_subgroupClusteredAdd2(weight); // 8 -> 4
        weight = __inline_subgroupShuffle(weight, sgInvocationID << 1);
        if (sgInvocationID < 4) sampler_buffer[superpixel_offset + 4 + sgInvocationID] = weight;
        weight = __inline_subgroupClusteredAdd2(weight); // 4 -> 2
        weight = __inline_subgroupShuffle(weight, sgInvocationID << 1);
        if (sgInvocationID < 2) sampler_buffer[superpixel_offset + 2 + sgInvocationID] = weight;
        weight = __inline_subgroupClusteredAdd2(weight); // 4 -> 2
        weight = __inline_subgroupShuffle(weight, sgInvocationID << 1);
        if (sgInvocationID < 1) sampler_buffer[superpixel_offset + 1 + sgInvocationID] = weight;
    }
}