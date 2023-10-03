#include "../../../include/common/space_filling_curve.hlsli"
#include "../../vxgi/include/vxgi_interface.hlsli"
#include "../include/vxguiding_interface.hlsli"
#include "shared.hlsli"

RWStructuredBuffer<TreeNode> u_Nodes;
RWStructuredBuffer<int> cluster_roots;
RWStructuredBuffer<int> compact2leaf;
RWStructuredBuffer<uint16_t2> u_Descendant;

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

const StructuredBuffer<uint64_t> u_Codes;
const StructuredBuffer<float> u_vxIrradiance;
const StructuredBuffer<uint> u_compactIndex;
const StructuredBuffer<int> u_clusterIndex;
const StructuredBuffer<TreeConstrIndirectArgs> u_ConstrIndirectArgs;

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(uint3 dtid: SV_DispatchThreadID) {
    const int numVXs = u_ConstrIndirectArgs[0].numValidVPLs;

    const int tid = int(dtid.x);
    const int numInternalNodes = numVXs - 1;
    const int numTotalNodes = numVXs * 2 - 1;

    if (tid < 32) {
        cluster_roots[tid] = -1;
    }

    if (tid >= numTotalNodes)
        return;

    // Init nodes
    TreeNode node;
    node.parent_idx = 0xFFFF;
    node.left_idx   = 0xFFFF;
    node.right_idx  = 0xFFFF;
    node.vx_idx     = 0xFFFF;
    node.flag = 0x0;
    node.intensity = 0.0f;
    node.aabbMax = uint2(0);
    node.aabbMin = uint2(0);

    if (tid < numInternalNodes) {
        u_Nodes[tid] = node;
        return;
    }
    
    // Init leaf nodes
    VoxelTexInfo info = CreateVoxelTexInfo(gVD);
    const uint leafID = tid - numInternalNodes;
    const uint compactID = uint(u_Codes[leafID] & 0XFFFF);
    const uint voxelID = u_compactIndex[compactID];
    const int3 mapPos = ReconstructIndex(voxelID, 64);
    const AABB aabb = VoxelToBound(mapPos, 0, info);
    const uint clusterID = u_clusterIndex[compactID];
    node.setAABBMin(aabb.min);
    node.setAABBMax(aabb.max);
    node.intensity = u_vxIrradiance[compactID];
    node.vx_idx = int16_t(compactID);
    node.flag = clusterID;
    u_Descendant[tid] = uint16_t2(uint16_t(leafID));
    compact2leaf[compactID] = tid;
    u_Nodes[tid] = node;
}
