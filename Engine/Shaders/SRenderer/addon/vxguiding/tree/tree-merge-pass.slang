#include "../../../include/common/space_filling_curve.hlsli"
#include "../../../include/common/geometry.hlsli"
#include "shared.hlsli"

RWStructuredBuffer<TreeNode> u_Nodes;
// RWStructuredBuffer<int> u_cluster_nodes;
StructuredBuffer<TreeConstrIndirectArgs> u_ConstrIndirectArgs;
RWStructuredBuffer<int> cluster_roots;

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(uint3 DTid: SV_DispatchThreadID) {
    uint numVPLs = u_ConstrIndirectArgs[0].numValidVPLs;
    if (DTid.x >= numVPLs) return;
    const int numInternalNodes = int(numVPLs - 1);
    const int idx = int(DTid.x + numInternalNodes);

    // Get first parent
    uint16_t parent = u_Nodes[idx].parent_idx;

    // Get node and vpl index for first child
    int lhs_node_id = u_Nodes[parent].left_idx;
    // Load first child data
    TreeNode lhs = u_Nodes[lhs_node_id];

    int i = 0;
    if (numVPLs == 1 && parent == 0xFFFF) {
        // this is the root node
        cluster_roots[u_Nodes[idx].flag] = 0;
        return;
    }
    while (parent != 0xFFFF) { // invalid index
        uint old;
        InterlockedCompareExchange(u_Nodes[parent].flag, 0u, 1u, old);
        if (old == 0) {
            // this is the first thread entered here.
            // wait the other thread from the other child node.
            return;
        }

        // Get child node indices
        const uint lidx = u_Nodes[parent].left_idx;
        const uint ridx = u_Nodes[parent].right_idx;

        // Determine vpl/node indices for right-hand-side
        const uint rhs_node_id = (lhs_node_id != ridx) ? ridx : lidx;
        // Load data of second child
        TreeNode rhs = u_Nodes[rhs_node_id];

        // Merge nodes
        TreeNode merged = u_Nodes[parent];
        merged.intensity = lhs.intensity + rhs.intensity;
        const float3 aabbMin = min(lhs.getAABBMin(), rhs.getAABBMin());
        const float3 aabbMax = max(lhs.getAABBMax(), rhs.getAABBMax());
        merged.setAABBMin(aabbMin);
        merged.setAABBMax(aabbMax);
        const bool cluster_split = lhs.flag != rhs.flag;
        merged.flag = (!cluster_split) ? lhs.flag : 0xffffffffu;
        if (lhs.flag != merged.flag && lhs.flag!= 0xffffffffu) {
            cluster_roots[lhs.flag] = lhs_node_id;
        }
        if (rhs.flag != merged.flag && rhs.flag != 0xffffffffu) {
            cluster_roots[rhs.flag] = rhs_node_id;
        }
        if (merged.parent_idx == 0xFFFF && lhs.flag == rhs.flag) {
            cluster_roots[merged.flag] = 0;
        }
        // Write merged vpl data
        u_Nodes[parent] = merged;

        // Save merged vpl for next round
        lhs       = merged;
        lhs_node_id = parent;

        // look for the next parent...
        parent = u_Nodes[parent].parent_idx;
    }
}