#include "seminee_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    uint numVPLs;
};

RWStructuredBuffer<TreeNode> u_Nodes;
RWStructuredBuffer<VPLMerge> u_Merge;
const RWStructuredBuffer<uint64_t> u_Codes;

RWStructuredBuffer<TreeConstrIndirectArgs> u_ConstrIndirectArgs;

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(uint3 dtid: SV_DispatchThreadID) {
    const uint tid = dtid.x;
    const int maxVPLs          = int(numVPLs);
    const int numInternalNodes = int(numVPLs) - 1;
    const int numTotalNodes    = int(numVPLs)*2 - 1;

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

    if (tid >= numVPLs) return;
    if (tid > 0 && u_Codes[tid] == uint64_t(-1) && u_Codes[tid - 1] != uint64_t(-1)) {
        int v_numVPLs = int(tid);
        int v_numInternalNodes = v_numVPLs - 1;
        int v_numTotalNodes = v_numVPLs * 2 - 1;
        u_ConstrIndirectArgs[0].dispatch_leaf = int3((v_numVPLs + 255) / 256, 1, 1);
        u_ConstrIndirectArgs[0].dispatch_internal = int3((v_numInternalNodes + 255) / 256, 1, 1);
        u_ConstrIndirectArgs[0].dispatch_node = int3((v_numTotalNodes + 255) / 256, 1, 1);
        u_ConstrIndirectArgs[0].numValidVPLs = tid;
        u_ConstrIndirectArgs[0].draw_rects = int4(6, v_numTotalNodes, 0, 0);
    }
}
