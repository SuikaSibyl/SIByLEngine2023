#include "../../vxgi/include/conetrace_utils.hlsli"
#include "../../vxgi/include/vxgi_interface.hlsli"

const StructuredBuffer<int> u_inverse_index;
RWStructuredBuffer<int> u_associate_buffer;
RWStructuredBuffer<uint> u_cluster_counter;    // 32
RWStructuredBuffer<float4> u_cluster_gathered; // 32 * (32 * 32)

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int3 Gid: SV_GroupID,
    int GidFlatten: SV_GroupIndex,
) {
    // Grab the bsdf sampled buffer
    const VoxelTexInfo info = CreateVoxelTexInfo(gVD);
    const float4 vpl_position = u_vpl_position[pixelID];
    if (all(vpl_position == float4(0))) return;
    
    const int3 vxID = int3(WorldToVoxel(vpl_position.xyz, info) * info.volumeDimension);
    const int compactID = u_inverse_index[vxID];
    const int clusterID = u_associate_buffer[compactID];
    if (clusterID == -1) return;

    // virtual point light injection
    uint clusterSubtaskIDFlatten;
    InterlockedAdd(u_cluster_counter[clusterID], 1, clusterSubtaskIDFlatten);
    if (clusterSubtaskIDFlatten < 1024) {
        u_cluster_gathered[clusterID * 1024 + clusterSubtaskIDFlatten] = vpl_position;
    }

    // Visibility injection
    InterlockedOr(u_spixel_visibility[spixelID2D], uint(1 << clusterID));
}