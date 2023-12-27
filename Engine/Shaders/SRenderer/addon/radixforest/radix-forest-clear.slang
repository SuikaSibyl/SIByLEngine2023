#include "radix-forest-sample.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int data_num;
};

// const StructuredBuffer<float> u_prefixsum;
RWStructuredBuffer<int> u_otherbounds;
RWStructuredBuffer<int2> u_nodes;

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    const int nodeId = int(dtid.x);
    if (nodeId >= data_num) return;
    // set all u_otherbounds to -1
    u_otherbounds[nodeId] = -1;
    // set all u_nodes to -1
    u_nodes[nodeId] = int2(-1, -1);
    // find the forset_table
}