#include "bitonic_common.hlsli"

#ifdef INDIRECT_DISPATCH
[[vk::push_constant]]
cbuffer PushConstantBuffer {
    int CounterOffset;
};
ByteAddressBuffer u_CounterBuffer;
#else
[[vk::push_constant]]
cbuffer PushConstantBuffer {
    uint NumElements;
};
#endif

RWStructuredBuffer<KEY_TYPE> g_SortBuffer;

groupshared uint64_t gs_SortValues[2048];

void LoadKeyIndexPair(uint Element, uint ListCount)  {
    uint64_t value = Element < ListCount ? g_SortBuffer[Element] : NULL_KEY;
    gs_SortValues[Element & 2047] = value;
}

void StoreKeyIndexPair(uint Element, uint ListCount) {
    if (Element < ListCount)
        g_SortBuffer[Element] = gs_SortValues[Element & 2047];
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void ComputeMain(uint3 gid: SV_GroupID, uint gi: SV_GroupIndex) {
    const uint GroupStart = gid.x * 2048;
#ifdef INDIRECT_DISPATCH
    const uint NumElements = u_CounterBuffer.Load(CounterOffset);
#endif

    // Load from memory into LDS to prepare sort
    LoadKeyIndexPair(GroupStart + gi, NumElements);
    LoadKeyIndexPair(GroupStart + gi + 1024, NumElements);

    GroupMemoryBarrierWithGroupSync();

    // This is better unrolled because it reduces ALU and because some
    // architectures can load/store two LDS items in a single instruction
    // as long as their separation is a compile-time constant.
    [unroll]
    for (uint j = 1024; j > 0; j /= 2) {
        uint Index2 = InsertOneBit(gi, j);
        uint Index1 = Index2 ^ j;

        uint64_t A = gs_SortValues[Index1];
        uint64_t B = gs_SortValues[Index2];

        if (ShouldSwap(A, B)) {
            // Swap
            gs_SortValues[Index1] = B;
            gs_SortValues[Index2] = A;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    StoreKeyIndexPair(GroupStart + gi, NumElements);
    StoreKeyIndexPair(GroupStart + gi + 1024, NumElements);
}
