#include "../../../include/common/cpp_compatible.hlsli"
#include "../../../include/common/random.hlsli"
#include "../../vxgi/include/vxgi_interface.hlsli"
#include "voxel_cluster.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    uint sample_batch;
    bool use_seed;
};

ByteAddressBuffer u_vxCounter;
StructuredBuffer<uint> u_pCompactIndex;
StructuredBuffer<int> u_seed;
StructuredBuffer<float4> u_vxNormal;

RWStructuredBuffer<svoxel_info> u_SVoxels;
RWStructuredBuffer<svoxel_info> u_accumSVoxels;

// 8x8x8: number of clusters in each voxel
RWTexture3D<uint> u_hashingCount;
// 8x8x8: indices of clusters in each hashing voxel
//        uint32 is decomposed into 4 x 8bit clusters
RWTexture3D<uint> u_hashingIndices;

RWStructuredBuffer<int4> u_indirectDispatchArgs;

groupshared uint gs_SpatialHashingCounts[8][8][8];
groupshared uint gs_SpatialHashingIndices[8][8][8];

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

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

    const int maxID = int(u_vxCounter.Load(0));
    if (Gid < 32) {
        int seedID;
        if (use_seed) {
            seedID = u_seed[Gid];
        }
        else {
            RandomSamplerState rng = InitRandomSampler(uint2(0, Gid), sample_batch, 1);
            seedID = clamp(int(GetNextRandom(rng) * maxID), 0, maxID - 1);
        }

        const uint voxelID = u_pCompactIndex[seedID];
        const int3 voxelPos = ReconstructIndex(voxelID, gVD.voxelSize);
        const float4 normal = u_vxNormal[seedID];

        svoxel_info voxel_info;
        voxel_info.center = float3(voxelPos);
        voxel_info.id = Gid;
        voxel_info.color_info = normal.xyz;
        voxel_info.no_voxels = 0;
        u_SVoxels[Gid] = voxel_info;
        
        empty_svoxel.id = Gid;
        u_accumSVoxels[Gid] = empty_svoxel;

        // Inject the new supervoxels positions into the hashing maps
        const int3 voxel_id = voxelPos / 8;
        uint slot_id = 0;
        InterlockedAdd(gs_SpatialHashingCounts[voxel_id.x][voxel_id.y][voxel_id.z], 1u, slot_id);
        if (slot_id < 4) {
            uint slot_offset = Gid << (slot_id * 8);
            InterlockedAdd(gs_SpatialHashingIndices[voxel_id.x][voxel_id.y][voxel_id.z], slot_offset);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    u_hashingCount[DTid.xyz] = gs_SpatialHashingCounts[DTid.x][DTid.y][DTid.z];
    u_hashingIndices[DTid.xyz] = gs_SpatialHashingIndices[DTid.x][DTid.y][DTid.z];

    if (Gid == 0) {
        u_indirectDispatchArgs[0] = int4((maxID + 255) / 256, 1, 1, 0);
        u_indirectDispatchArgs[1] = int4((maxID * 2 - 1 + 255) / 256, 1, 1, 0);
    }
}