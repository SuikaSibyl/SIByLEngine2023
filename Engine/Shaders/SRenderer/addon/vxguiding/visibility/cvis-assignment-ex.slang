#include "../../gSLICr/gSLICr_interface.hlsli"
#include "../../vxgi/include/conetrace_utils.hlsli"
#include "../../vxgi/include/vxgi_interface.hlsli"
#include "../geometry/common.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 map_size;
};

Texture2D<int> u_spixelIdx;
Texture2D<float4> u_vpl_position;
StructuredBuffer<uint> u_hashRecord;
StructuredBuffer<int> u_associate_buffer;
StructuredBuffer<int> u_inverse_index;

RWTexture2D<uint> u_spixel_visibility;  // 40 * 23
RWStructuredBuffer<uint> u_cluster_counter;    // 32
RWStructuredBuffer<float4> u_cluster_gathered; // 32 * (32 * 32)

uint PackInt2ToUInt(int2 xy) {
    return (uint(xy.x) << 16) | uint(xy.y);
}

groupshared uint gs_spixel_counter[9];
groupshared uint gs_spixel_offset[9];

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int3 Gid: SV_GroupID,
    int GidFlatten: SV_GroupIndex,
) {
    const int2 pixelID = DTid.xy;
    const int spixelID = u_spixelIdx[pixelID];
    const int2 spixelID2D = int2(spixelID % map_size.x, spixelID / map_size.x);

    // Grab the bsdf sampled buffer
    const VoxelTexInfo info = GetVoxelTexInfo();
    const float4 vpl_position = u_vpl_position[pixelID];
    if (all(vpl_position == float4(0))) return;

    const int3 vxID = int3(WorldToVoxel(vpl_position.xyz, info) * info.volumeDimension);
    uint32_t bucketID = QueryBucketID(vxID, u_config);
    uint32_t key = QueryKey(vxID);
    int cellID = FindCellID(bucketID, key, u_config, u_hashRecord);
    int globalID = GetGlobalID(bucketID, cellID, u_config);
    const int compactID = u_inverse_index[globalID];
    const int clusterID = u_associate_buffer[compactID];
    if (clusterID == -1) return;
    
    // virtual point llight injection
    uint clusterSubtaskIDFlatten;
    InterlockedAdd(u_cluster_counter[clusterID], 1, clusterSubtaskIDFlatten);
    if (clusterSubtaskIDFlatten < 1024) {
        u_cluster_gathered[clusterID * 1024 + clusterSubtaskIDFlatten] = vpl_position;
    }
    
    // Visibility injection
    InterlockedOr(u_spixel_visibility[spixelID2D], uint(1 << clusterID));
}