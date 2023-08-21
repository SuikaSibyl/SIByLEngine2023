#include "seminee_interface.hlsli"
#include "../../include/common/space_filling_curve.hlsli"

const RWStructuredBuffer<uint64_t>  u_Codes;
RWStructuredBuffer<TreeNode> u_Nodes;
RWStructuredBuffer<TreeConstrIndirectArgs> u_ConstrIndirectArgs;

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(uint3 tid: SV_DispatchThreadID) {
    uint numVPLs = u_ConstrIndirectArgs[0].numValidVPLs;
    uint thread_id = tid.x;
    if (thread_id >= numVPLs) return;
    
    const uint64_t m64 = u_Codes[thread_id];
    if (m64 == uint64_t(-1)) // Invalid VPL!
        return;

    const uint idx = uint(m64 & 0x1FFFFFF);

    // Assign vpl index to leaf node (node buffer : [internal nodes, leaf nodes])
    const uint numInternalNodes = numVPLs - 1;
    u_Nodes[thread_id + numInternalNodes].vpl_idx = idx;
}