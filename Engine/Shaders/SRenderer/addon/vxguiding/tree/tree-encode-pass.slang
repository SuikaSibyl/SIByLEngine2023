#include "../../../include/common/space_filling_curve.hlsli"
#include "../../vxgi/include/vxgi_interface.hlsli"
#include "../include/vxguiding_interface.hlsli"
#include "shared.hlsli"

RWStructuredBuffer<uint64_t> u_Codes;
RWStructuredBuffer<TreeConstrIndirectArgs> u_ConstrIndirectArgs;

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

ByteAddressBuffer u_vxCounter;
StructuredBuffer<uint> u_compactIndex;
StructuredBuffer<int> u_clusterIndex;

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(uint3 dtid: SV_DispatchThreadID) {
    const int numVXs = u_vxCounter.Load(0);
    const int tid = int(dtid.x);
    const int numInternalNodes = numVXs - 1;
    const int numTotalNodes = numVXs * 2 - 1;

    if (tid >= numVXs)
        return;

    // Init leaf nodes
    VoxelTexInfo info = CreateVoxelTexInfo(gVD);
    const uint leafID = tid;
    const uint voxelID = u_compactIndex[leafID];
    const int3 mapPos = ReconstructIndex(voxelID, gVD.voxelSize);
    const AABB aabb = VoxelToBound(mapPos, 0, info);
    const uint clusterID = u_clusterIndex[leafID];

    // Also write codes
    const float3 unipos = clamp(mapPos + 0.5, 0, gVD.voxelSize) / gVD.voxelSize;
    const uint64_t posCode = ZCurve3DToMortonCode(unipos);
    const uint64_t clusterCode = clusterID;
    const uint64_t idCode = leafID;
    u_Codes[leafID] = (clusterCode << 48) | (posCode << 16) | (idCode << 0);

    if (tid == 0) {
        // tid == 0, write indirect args
        u_ConstrIndirectArgs[0].dispatch_leaf = int3((numVXs + 255) / 256, 1, 1);
        u_ConstrIndirectArgs[0].dispatch_internal = int3((numInternalNodes + 255) / 256, 1, 1);
        u_ConstrIndirectArgs[0].dispatch_node = int3((numTotalNodes + 255) / 256, 1, 1);
        u_ConstrIndirectArgs[0].numValidVPLs = numVXs;
        u_ConstrIndirectArgs[0].draw_rects = int4(6, numTotalNodes, 0, 0);
    }
}
