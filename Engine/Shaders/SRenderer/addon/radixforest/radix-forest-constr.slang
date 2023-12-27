#include "radix-forest-sample.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int data_num;
    int partition_num;
};

const StructuredBuffer<float> u_pmf;      // prefix sum of pmf
RWStructuredBuffer<int> u_otherbounds;
RWStructuredBuffer<int2> u_nodes;
RWStructuredBuffer<int> u_hashtable;

float load_data(int index) {
    if (index < 0 || index >= data_num) return 1.f;
    return u_pmf[index];
}

uint as_uint(float x) {
    return uint(double(x) * double(4294967295u));
}

int pack_leaf(int id) {
    return 0x80000000 | id;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    int nodeId = int(dtid.x);
    if (nodeId >= data_num) return;
    const int curCell = int(floor(u_pmf[nodeId] * partition_num));
    int2 range = int2(nodeId, nodeId);
    int other_bound = -1;

    bool isLeaf = true;
    bool isInterval = false;
    
    int loop = 0;
    while (true) {
        const float value_low = load_data(range.x);
        const float value_high = load_data(range.y);
        float value_neighbor_low = load_data(range.x - 1);
        float value_neighbor_high = load_data(range.y + 1);
        if (floor(value_neighbor_low * partition_num) < curCell)
            value_neighbor_low = 1;
        if (floor(value_neighbor_high * partition_num) > curCell)
            value_neighbor_high = 1;
        const bool is_internal = (value_neighbor_low == 1 && value_neighbor_high == 1);
        int child = ((as_uint(value_low) ^ as_uint(value_neighbor_low)) 
            > (as_uint(value_high) ^ as_uint(value_neighbor_high))) ? 0 : 1;
        if (is_internal & !isLeaf) child = 1;

        const int parent = (child == 0) ? (range.y + 1) : range.x;
        u_nodes[parent][child] = isLeaf ? pack_leaf(nodeId) : nodeId;
        
        // break when merging with the node from other threads
        InterlockedExchange(u_otherbounds[parent], range[child], other_bound);
        // should exit when both neighbor exceed the cell

        if (is_internal) {
            isInterval = true;
            u_nodes[parent][0] = pack_leaf(range.x - 1);
        }
        
        if (is_internal & !isLeaf) {
            break;
        }
        
        if (other_bound != -1) {
            range[1 - child] = other_bound;
            nodeId = parent;
        } else break;

        isLeaf = false;

        loop++;
        if (loop > 100)
            break;
    }

    if (isInterval) {
        const float value_neighbor_high = load_data(range.y + 1);
        const int nextCell = int(floor(value_neighbor_high * partition_num));
        for (int i = curCell; i < nextCell; ++i) {
            u_hashtable[i] = nodeId;
        }
    }
}