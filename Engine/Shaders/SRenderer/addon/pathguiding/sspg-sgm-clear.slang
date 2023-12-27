[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    bool clearAll;
};

RWTexture2D<float4> u_gmmStatisticsPack0;
RWTexture2D<float4> u_gmmStatisticsPack1;
RWTexture2D<float4> u_gmmStatisticsPack0Prev;
RWTexture2D<float4> u_gmmStatisticsPack1Prev;

[shader("compute")]
[numthreads(16, 16, 1)] // 8x8x8 threads in a group
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    if (any(dtid.xy >= resolution)) return;

    if (clearAll) {
        u_gmmStatisticsPack0[dtid.xy] = float4(0, 0, 0, 0);
        u_gmmStatisticsPack1[dtid.xy] = float4(0, 0, 0, 0);
        u_gmmStatisticsPack0Prev[dtid.xy] = float4(0, 0, 0, 0);
        u_gmmStatisticsPack1Prev[dtid.xy] = float4(0, 0, 0, 0);
    }
}