[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    int lobeCount;
};

const StructuredBuffer<float4> u_gmmStatisticsPack0;
const StructuredBuffer<float4> u_gmmStatisticsPack1;
RWStructuredBuffer<float4> u_gmmStatisticsPack0Prev;
RWStructuredBuffer<float4> u_gmmStatisticsPack1Prev;

[shader("compute")]
[numthreads(16, 16, 1)] // 8x8x8 threads in a group
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    if (any(dtid.xy >= resolution)) return;
    
    const int index = (dtid.y * resolution.x + dtid.x) * lobeCount;
    for (int i = 0; i < lobeCount; ++i) {
        u_gmmStatisticsPack0Prev[index + i] = u_gmmStatisticsPack0[index + i];
        u_gmmStatisticsPack1Prev[index + i] = u_gmmStatisticsPack1[index + i];
    }
}