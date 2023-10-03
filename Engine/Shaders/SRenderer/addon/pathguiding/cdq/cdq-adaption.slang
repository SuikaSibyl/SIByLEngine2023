#include "../../../include/common/cylindrical.hlsli"
#include "compressed-directional-quadtree.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    bool initialize;
    bool adaption;
};

const Texture2D<float4> u_PreSampleList;
RWStructuredBuffer<uint64_t> u_cdq;

groupshared uint gs_InnerNodeContribution[64];
groupshared uint gs_LeaveNodeContribution[64];

groupshared int gs_MinInnerNode;
groupshared int gs_MaxLeaveNode;

void SplatContribution(
    in_ref(uint64_t) cdq,
    in_ref(float2) uv,
    in_ref(uint) quantized_contribution,
) {
    uint child_id = find_child_id(uv);
    uint nodeID = child_id;
    int prevNodeID = -1;
    while (nodeID < 64 && is_inner(cdq, nodeID)) {
        uv = zoom_in_coord(uv);
        child_id = find_child_id(uv);
        prevNodeID = nodeID;
        nodeID = child(cdq, nodeID, child_id);
    }
    // is leave node
    if (prevNodeID != -1) {
        uint first_nodeID = child(cdq, prevNodeID, 0);
        if ((cdq & (set_leftmost_n_bits_64(4) >> first_nodeID)) == 0) {
            InterlockedAdd(gs_InnerNodeContribution[prevNodeID], quantized_contribution);
        }
    }
    InterlockedAdd(gs_LeaveNodeContribution[nodeID], quantized_contribution);
}

uint quantize_contribution(in_ref(float) contribution) {
    return uint(contribution * 10000);
}

[shader("compute")]
[numthreads(16, 16, 1)] // 8x8x8 threads in a group
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int gid: SV_GroupIndex
) {
    if (!adaption) return;

    // Clear group shared memory
    if (gid < 64) {
        gs_InnerNodeContribution[gid] = 0;
        gs_LeaveNodeContribution[gid] = 0;
        gs_MinInnerNode = -1;
        gs_MaxLeaveNode = -1;
    }
    GroupMemoryBarrierWithGroupSync();

    // Load data
    const int2 sampleID = dtid.xy;
    const float4 sample = u_PreSampleList[sampleID];
    const float3 direction = sample.xyz;
    const float contribution = sample.w;
    const float2 cylcoord = CylindricalToUnit(UnitVectorToCylindrical(direction));

    // const float2 cylcoord = float2(gtid.xy + 0.5) / 16.0;
    // float contribution = cylcoord.y;
    // contribution = contribution * contribution;

    uint64_t cdq = initialize ? 0xfffe000000000000ull : u_cdq[0];
    if (countbits(uint(cdq >> 32) + countbits(uint(cdq & 0xffffffff))) != 15) {
        cdq = 0xfffe000000000000ull;
    }

    // splat contribution
    if (!all(sample == float4(0)))
        SplatContribution(cdq, cylcoord, quantize_contribution(contribution));

    GroupMemoryBarrierWithGroupSync();

    if (gid < 64) {
        if (is_inner(cdq, gid)) {
            uint first_nodeID = child(cdq, gid, 0);
            if ((cdq & (set_leftmost_n_bits_64(4) >> first_nodeID)) != 0) {
                gs_InnerNodeContribution[gid] = 0xffffffff;
            }
        }
        else {
            gs_InnerNodeContribution[gid] = 0xffffffff;
        }
    }

    if (gid >= WaveGetLaneCount()) {
        return;
    }

    {   // Get the inner node with minimum contribution
        const uint data0 = gs_InnerNodeContribution[gid];
        const uint data1 = gs_InnerNodeContribution[gid + 32];
        const uint min_data = min(data0, data1);
        const uint min_data_subgroup = WaveActiveMin(min_data);
        if ((min_data_subgroup == min_data) && (min_data != 0xffffffff)) {
            int min_inner_node;
            if (min_data == data0) {
                min_inner_node = gid;
            } else {
                min_inner_node = gid + 32;
            }
            gs_MinInnerNode = min_inner_node;
        }
    }
    {   // Get the leave node with maximum contribution
        const uint data0 = gs_LeaveNodeContribution[gid];
        const uint data1 = gs_LeaveNodeContribution[gid + 32];
        const uint max_data = max(data0, data1);
        const uint max_data_subgroup = WaveActiveMax(max_data);
        if ((max_data_subgroup == max_data) && (max_data != 0)) {
            int max_leave_node;
            if (max_data == data0) {
                max_leave_node = gid;
            } else {
                max_leave_node = gid + 32;
            }
            gs_MaxLeaveNode = max_leave_node;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (WaveIsFirstLane()) {
        const int min_inner_node = gs_MinInnerNode;
        const int max_leave_node = gs_MaxLeaveNode;

        u_cdq[1] = min_inner_node;
        u_cdq[2] = max_leave_node;

        if (min_inner_node == -1 || max_leave_node == -1) {
            u_cdq[0] = cdq;
            u_cdq[3] = 999;
            return;
        }
        
        uint64_t tmp = cdq;
        
        const uint inner_child = child(cdq, min_inner_node, 0);
        const uint leave_child = child(cdq, max_leave_node, 0);
        cdq = flip(cdq, min_inner_node);
        cdq = flip(cdq, max_leave_node);
        { // remove the inner node
            uint64_t mask_left = set_leftmost_n_bits_64(inner_child);
            uint64_t left_part = cdq & mask_left;
            uint64_t mask_right = set_rightmost_n_bits_64(64 - inner_child - 4);
            uint64_t right_part = cdq & mask_right;
            cdq = left_part | (right_part << 4);
        }
        { // inject the leave node
            uint64_t mask_left = set_leftmost_n_bits_64(leave_child);
            uint64_t left_part = cdq & mask_left;
            uint64_t mask_right = set_rightmost_n_bits_64(64 - leave_child);
            uint64_t right_part = cdq & mask_right;
            cdq = left_part | (right_part >> 4);
        }

        if (countbits(uint(cdq >> 32)) + countbits(uint(cdq & 0xffffffff)) == 15) {
            u_cdq[3] = 99999;
            u_cdq[0] = cdq;
        }
        else {
            u_cdq[4] = tmp;
            u_cdq[5] = cdq;
            u_cdq[3] = 9999;
        }
    }
}