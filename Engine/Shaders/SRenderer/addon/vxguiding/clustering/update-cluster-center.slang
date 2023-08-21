#include "../../../include/common/cpp_compatible.hlsli"
#include "../../vxgi/include/vxgi_interface.hlsli"
#include "voxel_cluster.hlsli"

RWStructuredBuffer<svoxel_info> u_SVoxels;
RWStructuredBuffer<svoxel_info> u_accumSVoxels;
// // 8x8x8: number of clusters in each voxel
RWTexture3D<uint> u_hashingCount;
// // 8x8x8: indices of clusters in each hashing voxel
// //        uint32 is decomposed into 4 x 8bit clusters
RWTexture3D<uint> u_hashingIndices;

groupshared uint gs_SpatialHashingCounts[8][8][8];
groupshared uint gs_SpatialHashingIndices[8][8][8];

/**
 */
[shader("compute")]
[numthreads(8, 8, 8)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int Gid: SV_GroupIndex
) {
    gs_SpatialHashingCounts[DTid.x][DTid.y][DTid.z] = 0;
    gs_SpatialHashingIndices[DTid.x][DTid.y][DTid.z] = 0;
    GroupMemoryBarrierWithGroupSync();

    svoxel_info empty_svoxel;
    empty_svoxel.center = float3(0.0f);
    empty_svoxel.color_info = float3(0.0f);
    empty_svoxel.no_voxels = 0;

    const int spixel_idx = Gid;
    if (Gid < 32) {
        // Update the avg supervoxels by accumulated supervoxels
        const svoxel_info accum_info = u_accumSVoxels[spixel_idx];
        
        if (accum_info.no_voxels > 0) {
            svoxel_info avg_info;
            avg_info.center = accum_info.center / accum_info.no_voxels;
            avg_info.color_info = accum_info.color_info / accum_info.no_voxels;
            avg_info.no_voxels = accum_info.no_voxels;
            avg_info.id = accum_info.id;
            u_SVoxels[spixel_idx] = avg_info;

            // Inject the new supervoxels positions into the hashing maps
            const int3 voxel_id = int3(avg_info.center / 8);
            uint slot_id = 0;
            InterlockedAdd(gs_SpatialHashingCounts[voxel_id.x][voxel_id.y][voxel_id.z], 1u, slot_id);
            if (slot_id < 4) {
                uint slot_offset = Gid << (slot_id * 8);
                InterlockedAdd(gs_SpatialHashingIndices[voxel_id.x][voxel_id.y][voxel_id.z], slot_offset);
            }
        }
    }

    DeviceMemoryBarrierWithGroupSync();
    
    if (Gid < 32)
        u_accumSVoxels[spixel_idx] = empty_svoxel;

    GroupMemoryBarrierWithGroupSync();

    u_hashingCount[DTid.xyz] = gs_SpatialHashingCounts[DTid.x][DTid.y][DTid.z];
    u_hashingIndices[DTid.xyz] = gs_SpatialHashingIndices[DTid.x][DTid.y][DTid.z];
}