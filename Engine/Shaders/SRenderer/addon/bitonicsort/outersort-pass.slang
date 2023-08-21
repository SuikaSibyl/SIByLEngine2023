#include "bitonic_common.hlsli"

#ifdef INDIRECT_DISPATCH
[[vk::push_constant]]
cbuffer PushConstantBuffer {
    uint k; // k >= 4096
    uint j; // j >= 2048 && j < k
    int CounterOffset;
};
ByteAddressBuffer u_CounterBuffer;
#else
[[vk::push_constant]]
cbuffer PushConstantBuffer {
    uint k; // k >= 4096
    uint j; // j >= 2048 && j < k
    uint NumElements;
};
#endif

RWStructuredBuffer<KEY_TYPE> g_SortBuffer;

[shader("compute")]
[numthreads(1024, 1, 1)]
void ComputeMain(uint3 tid: SV_DispatchThreadID) {
#ifdef INDIRECT_DISPATCH
    const uint NumElements = u_CounterBuffer.Load(CounterOffset);
#endif
    // Form unique index pair from dispatch thread ID
    uint Index2 = InsertOneBit(tid.x, j);
    uint Index1 = Index2 ^ (k == 2 * j ? k - 1 : j);

    if (Index2 >= NumElements)
        return;
    
    KEY_TYPE A = g_SortBuffer[Index1];
    KEY_TYPE B = g_SortBuffer[Index2];

    if (ShouldSwap(A, B)) {
        g_SortBuffer[Index1] = B;
        g_SortBuffer[Index2] = A;
    }
}
