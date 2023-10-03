#include "sst-interface.hlsli"

RWStructuredBuffer<TreeNode> u_Nodes;
RWStructuredBuffer<VPLMerge> u_Merge;

const StructuredBuffer<TreeConstrIndirectArgs> u_ConstrIndirectArgs;

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(uint3 dtid: SV_DispatchThreadID) {
    const uint tid = dtid.x;
    const int maxVPLs = int(u_ConstrIndirectArgs[0].numValidVPLs);
    const int numTotalNodes = maxVPLs * 2 - 1;

    if (tid >= numTotalNodes)
        return;
    
    // Init nodes
    TreeNode node;
    node.parent_idx = 0xFFFFFFFF;
    node.left_idx   = 0xFFFFFFFF;
    node.right_idx  = 0xFFFFFFFF;
    node.vpl_idx    = 0xFFFFFFFF;
    node.flag = 0x0;
    u_Nodes[tid]  = node;
    
    // Init merge data
    VPLMerge merge;
    merge.ApproxScore = float2(1.f, 0.f); // Normal score / Normal Z std
    u_Merge[tid] = merge;
}
