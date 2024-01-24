#include "../../vxgi/include/conetrace_utils.hlsli"
#include "cluster-common.hlsli"

RWStructuredBuffer<int> u_Clusters;

const StructuredBuffer<uint> u_CompactIndices;
const StructuredBuffer<uint4> u_IndirectArgs;
const StructuredBuffer<uint4> u_RowVisibility;
const StructuredBuffer<svoxel_info> u_ClusterInfo;
const StructuredBuffer<float> u_PremulIrradiance;

#define DESC_BUFFER u_RowVisibility
#define GET_DESC(i) DESC_BUFFER[i]

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(int3 DTid: SV_DispatchThreadID) {
    const uint counterCount = u_IndirectArgs[0].w;
    const int compactID = int(DTid.x);
    if(compactID >= counterCount) return;

    const int voxelID = int(u_CompactIndices[compactID]);
    const int3 voxelPos = ReconstructIndex(voxelID, gVD.voxelSize);
    const int3 gridPos = voxelPos / 8;
    const float intensity = u_PremulIrradiance[compactID];

    // find the closet cluster ID
    const int clusterCount = 32;
    const float dist_normalizer = 1.0f / (1.732f * 8);
    const DESC_TYPE desc = GET_DESC(compactID);
    const float3 vPos = float3(voxelPos);
    int minidx = -1;
    float dist = 999999.9999f;
    for(int i = 0; i < clusterCount; ++i){
        const int clusterID = i;
        svoxel_info cluster_info = u_ClusterInfo[clusterID];
        const DESC_TYPE clusterDesc = cluster_info.desc_info; // TODO
        const float3 clusterPos = cluster_info.center;
        const float wdist = ComputeDistance(clusterDesc, desc, true,
                                            float3(voxelPos), cluster_info.center,
                                            intensity, cluster_info.intensity,
                                            0.0f, 1.f);
        if (wdist < dist) {
            dist = wdist;
            minidx = clusterID;
        }
    }

    u_Clusters[compactID] = minidx;
}