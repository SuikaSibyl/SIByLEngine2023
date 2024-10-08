#include "../../../include/common/random.hlsli"
#include "../../vxgi/include/conetrace_utils.hlsli"
#include "cluster-common.hlsli"

RWStructuredBuffer<int> u_Seeds;
RWStructuredBuffer<svoxel_info> u_RowClusterInfo;

const StructuredBuffer<uint4> u_IndirectArgs;
const StructuredBuffer<uint4> u_RowVisibility;
const StructuredBuffer<uint> u_CompactIndices;
const StructuredBuffer<float> u_PremulIrradiance;

#ifndef DESC_TYPE
#define DESC_TYPE uint4
#endif

groupshared float warp_prob[32];
groupshared DESC_TYPE current_center;
groupshared float3 current_center_pos;
groupshared float current_center_intensity;
groupshared int selected_cluster;

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void ComputeMain(int3 DTid: SV_DispatchThreadID) {
    const uint tid = DTid.x;
    const uint sgid = WaveGetLaneIndex();
    const uint sg = tid / 32;
    const int element_count = u_IndirectArgs[0].w;

    RandomSamplerState RNG = InitRandomSampler(uint2(0, tid), 0);
    const int rnd_element = clamp(int(GetNextRandom(RNG) * element_count), 0, element_count - 1);

    const int voxelID = int(u_CompactIndices[rnd_element]);
    const float3 position = float3(ReconstructIndex(voxelID, gVD.voxelSize));
    const DESC_TYPE element = u_RowVisibility[rnd_element];
    const float intensity = u_PremulIrradiance[rnd_element];
    
    if (tid == 0) {
        // random choose the first seed
        current_center = element;
        current_center_pos = position;
        current_center_intensity = intensity;
        u_Seeds[0] = rnd_element;
        u_RowClusterInfo[0] = svoxel_info(element, position, intensity);
    }
    if (sg == 0) {
        // reset the prob tree
        warp_prob[sgid] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    
    float dist = 100000000.f;
    
    for (int seed_id = 1; seed_id < 32; ++seed_id) {
        // calculate the distance to the nearest center
        const DESC_TYPE cluster_element = current_center;
        const float3 cluster_center_pos = current_center_pos;
        const float cluster_center_intensity = current_center_intensity;

        const float d = ComputeDistance(element, cluster_element, true,
                                        position, cluster_center_pos,
                                        intensity, cluster_center_intensity,
                                        0.f, 1.f);
        dist = min(dist, d);
        
        // calculate the prob of being selected
        float weight = dist * dist;
        const float warp_weight_sum = WaveActiveSum(weight);
        if (WaveIsFirstLane())
            warp_prob[sg] = warp_weight_sum;

        GroupMemoryBarrierWithGroupSync();

        if (sg == 0) {
            weight = warp_prob[sgid];
            if (WaveIsFirstLane()) weight = 0.f;
        }
        
        // build up the sum of prob tree
        // store 5 floats per thread instead of 64 floats per thread ...
        // not sure wheter using shared memory is faster or not
        float prob0_hierarchy[6];
        for (int i = 0; i < 5; ++i) {
            const float weight_neighbor = WaveShuffle(weight, sgid ^ (1 << i));
            const float weight_sum = weight + weight_neighbor;
            prob0_hierarchy[4 - i] = (weight_sum == 0) ? 0.5f : weight / weight_sum;
            weight = weight_sum;
        }

        // sample the tree
        float rnd = GetNextRandom(RNG);
        int nid = 0;
        for (int i = 0; i < 5; ++i) {
            const float prob0 = prob0_hierarchy[i];
            if (rnd < prob0) {
                rnd /= prob0;
            } else {
                nid += (16 >> i);
                rnd = (rnd - prob0) / (1 - prob0);
            }
            prob0_hierarchy[i + 1] = WaveShuffle(prob0_hierarchy[i + 1], nid);
        }

        if (tid == 0)
            selected_cluster = nid;

        GroupMemoryBarrierWithGroupSync();

        int selected_subid = WaveReadLaneAt(nid, 0);
        if (sg == selected_cluster && sgid == selected_subid) {
            u_Seeds[seed_id] = rnd_element;
            u_RowClusterInfo[seed_id] = svoxel_info(element, position, intensity);
            current_center = element;
        }

        GroupMemoryBarrierWithGroupSync();
    }
}