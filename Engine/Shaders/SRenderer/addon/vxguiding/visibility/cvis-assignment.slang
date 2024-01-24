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
RWStructuredBuffer<float4> u_cluster_gathered; // 32 * (32 * 32)

RWStructuredBuffer<int> u_associate_buffer;
Texture3D<int> u_inverse_index;

RWStructuredBuffer<float4> u_RepresentPixel;
RWStructuredBuffer<float4> u_RepresentVPL;

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

    if (GidFlatten < 9) {
        gs_spixel_counter[GidFlatten] = 0;
        gs_spixel_offset[GidFlatten] = 0;
    }

    const int2 spixelID2D = int2(spixelID % map_size.x, spixelID / map_size.x);
    const int2 ctr = pixelID / 32; // spixelID2D should be 9 neighbors of ctr
    const int2 offset_sp = (spixelID2D - ctr + int2(1, 1));
    const int gs_sp_offset = offset_sp.x + offset_sp.y * 3;

    GroupMemoryBarrierWithGroupSync();

    uint subtaskIDFlatten_gs;
    if (spixelID != -1) {
        InterlockedAdd(gs_spixel_counter[gs_sp_offset], 1, subtaskIDFlatten_gs);
    }

    GroupMemoryBarrierWithGroupSync();

    if (GidFlatten < 9) {
        const int gs_sp_offset = GidFlatten;
        const int2 offset_sp = int2(gs_sp_offset % 3, gs_sp_offset / 3) - int2(1, 1);
        const int2 spixelID2D = ctr + offset_sp;
        if (all(spixelID2D >= 0 && spixelID2D < map_size)) {
            InterlockedAdd(u_spixel_counter[spixelID2D], gs_spixel_counter[gs_sp_offset], gs_spixel_offset[gs_sp_offset]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (spixelID != -1) {
        const int2 taskOffset = spixelID2D * 32;
        uint subtaskIDFlatten;
        // InterlockedAdd(u_spixel_counter[spixelID2D], 1, subtaskIDFlatten);
        subtaskIDFlatten = gs_spixel_offset[gs_sp_offset] + subtaskIDFlatten_gs;
        if (subtaskIDFlatten < 1024) {
            const int2 subtaskID = int2(subtaskIDFlatten % 32, subtaskIDFlatten / 32);
            u_spixel_gathered[taskOffset + subtaskID] = pixelID;
        }
    }

    // Grab the bsdf sampled buffer
    const VoxelTexInfo info = CreateVoxelTexInfo(gVD);
    const float4 vpl_position = u_vpl_position[pixelID];
    if (all(vpl_position == float4(0))) return;

    const int3 vxID = int3(WorldToVoxel(vpl_position.xyz, info) * info.volumeDimension);
    const int compactID = u_inverse_index[vxID];
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