#include "../../gSLICr/gSLICr_interface.hlsli"
#include "../../vxgi/include/conetrace_utils.hlsli"
#include "../../vxgi/include/vxgi_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 map_size;
};

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

Texture2D<int> u_spixelIdx;
Texture2D<float4> u_vpl_position;

RWTexture2D<int2> u_spixel_gathered;    // 1280 * 736 = (40*32) * (23*32)
RWTexture2D<uint> u_spixel_counter;     // 40 * 23
RWTexture2D<uint> u_spixel_visibility;  // 40 * 23
RWStructuredBuffer<uint> u_cluster_counter;    // 32
RWStructuredBuffer<float4> u_cluster_gathered;  // 32 * (32 * 32)

RWStructuredBuffer<int> u_associate_buffer;
Texture3D<int> u_inverse_index;

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
    const int spixelID = u_spixelIdx[pixelID];
    if (spixelID == -1) return;

    const int2 spixelID2D = int2(spixelID % map_size.x, spixelID / map_size.x);
    const int2 taskOffset = spixelID2D * 32;

    uint subtaskIDFlatten;
    InterlockedAdd(u_spixel_counter[spixelID2D], 1, subtaskIDFlatten);
    if (subtaskIDFlatten < 1024) {
        const int2 subtaskID = int2(subtaskIDFlatten % 32, subtaskIDFlatten / 32);
        u_spixel_gathered[taskOffset + subtaskID] = pixelID;
    }

    // Grab the bsdf sampled buffer
    const VoxelTexInfo info = CreateVoxelTexInfo(gVD);
    const float4 vpl_position = u_vpl_position[pixelID];
    if (all(vpl_position == float4(0))) return;

    const int3 vxID = int3(WorldToVoxel(vpl_position.xyz, info) * info.volumeDimension);
    const int clusterID = u_associate_buffer[u_inverse_index[vxID]];
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