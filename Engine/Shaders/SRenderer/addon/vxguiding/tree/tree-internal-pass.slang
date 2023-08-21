
#include "../../../include/common/space_filling_curve.hlsli"
#include "shared.hlsli"

const StructuredBuffer<uint64_t> u_Codes;
const StructuredBuffer<TreeConstrIndirectArgs> u_ConstrIndirectArgs;
RWStructuredBuffer<TreeNode> u_Nodes;

int clz(uint x) {
    return 31 - int(firstbithigh(x));
}

int clz64(uint64_t x) {
    const int h = clz((uint) (x >> 32));
    const int l = clz((uint) (x & 0xFFFFFFFF));
    return h == 32 ? h + l : h;
}

void swap(inout uint u1, inout uint u2) {
    uint tmp = u1;
    u1 = u2;
    u2 = tmp;
}

int common_upper_bits(const uint64_t lhs, const uint64_t rhs) {
    return clz64(lhs ^ rhs);
}

uint2 determine_range(const uint num_leaves, uint idx) {
    if (idx == 0)
        return uint2(0, num_leaves - 1);

    // determine direction of the range
    const uint64_t self_code = u_Codes[idx];
    const int L_delta = common_upper_bits(self_code, u_Codes[idx - 1]);
    const int R_delta = common_upper_bits(self_code, u_Codes[idx + 1]);
    const int d = (R_delta > L_delta) ? 1 : -1;

    // Compute upper bound for the length of the range

    const int delta_min = min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = int(idx) + d * l_max;
    if (0 <= i_tmp && i_tmp < num_leaves) {
        delta = common_upper_bits(self_code, u_Codes[i_tmp]);
    }
    while (delta > delta_min) {
        l_max <<= 1;
        i_tmp = int(idx) + d * l_max;
        delta = -1;
        if (0 <= i_tmp && i_tmp < num_leaves) {
            delta = common_upper_bits(self_code, u_Codes[i_tmp]);
        }
    }

    // Find the other end by binary search
    int l = 0;
    int t = l_max >> 1;
    while (t > 0) {
        i_tmp = int(idx) + (l + t) * d;
        delta = -1;
        if (0 <= i_tmp && i_tmp < num_leaves) {
            delta = common_upper_bits(self_code, u_Codes[i_tmp]);
        }
        if (delta > delta_min) {
            l += t;
        }
        t >>= 1;
    }
    uint jdx = idx + l * d;
    if (d < 0) {
        swap(idx, jdx); // make it sure that idx < jdx
    }
    return uint2(idx, jdx);
}

uint find_split(const uint first, const uint last) {
    const uint64_t first_code = u_Codes[first];
    const uint64_t last_code = u_Codes[last];
    if (first_code == last_code) {
        return (first + last) >> 1;
    }
    const int delta_node = common_upper_bits(first_code, last_code);
    
    // binary search...
    int split = int(first);
    int stride = int(last - first);
    do {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last)
        {
            const int delta = common_upper_bits(first_code, u_Codes[middle]);
            if (delta > delta_node)
            {
                split = middle;
            }
        }
    } while (stride > 1);

    return split;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(uint3 DTid: SV_DispatchThreadID) {
    const uint numVPLs = u_ConstrIndirectArgs[0].numValidVPLs;
    const int idx = int(DTid.x);
    const int num_objects = int(numVPLs);

    if (idx >= max(0, num_objects - 1))
        return;

    // For all internal nodes, determine the range of objects they represent
    const uint2 ij = determine_range(num_objects, idx);
    const int gamma = int(find_split(ij.x, ij.y));

    u_Nodes[idx].left_idx = uint16_t(gamma);
    u_Nodes[idx].right_idx = uint16_t(gamma + 1);

    if (min(ij.x, ij.y) == gamma)
        u_Nodes[idx].left_idx += uint16_t(numVPLs - 1);
    if (max(ij.x, ij.y) == gamma + 1)
        u_Nodes[idx].right_idx += uint16_t(numVPLs - 1);

    u_Nodes[u_Nodes[idx].left_idx].parent_idx = uint16_t(idx);
    u_Nodes[u_Nodes[idx].right_idx].parent_idx = uint16_t(idx);
}
