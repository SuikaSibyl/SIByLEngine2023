[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    int lobeCount;
    bool clearAll;
};

RWStructuredBuffer<float4> u_gmmStatisticsPack0;
RWStructuredBuffer<float4> u_gmmStatisticsPack1;
RWStructuredBuffer<float4> u_gmmStatisticsPack0Prev;
RWStructuredBuffer<float4> u_gmmStatisticsPack1Prev;

[shader("compute")]
[numthreads(16, 16, 1)] // 8x8x8 threads in a group
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    if (any(dtid.xy >= resolution)) return;

    const int index = (dtid.y * resolution.x + dtid.x) * lobeCount;
    if (clearAll) 
    {
        float4 weights = float4(0.1, 0.2, 0.3, 0.4);
        for (int i = 0; i < lobeCount; ++i) {
            const float2 uv = float2(1.f) + float2(1.f) * float2(i / 2, i % 2);
            const float weight = i == 0 ? 1.f : 0.f;
            u_gmmStatisticsPack0[index + i] = float4(uv, 0.01, 0.01);
            u_gmmStatisticsPack1[index + i] = float4(0, 1.f, 0, weights[i]);
            u_gmmStatisticsPack0Prev[index + i] = float4(uv, 0.01, 0.01);
            u_gmmStatisticsPack1Prev[index + i] = float4(0, 1.f, 0, weights[i]);
        }
    }
}