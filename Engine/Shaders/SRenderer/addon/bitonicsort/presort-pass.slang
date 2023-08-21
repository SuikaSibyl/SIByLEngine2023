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

groupshared KEY_TYPE gs_SortValues[2048];

void LoadElement(uint index, uint ListCount) {
    if (index < ListCount) {
        KEY_TYPE value = g_SortBuffer[index];
        gs_SortValues[index & 2047] = value;
    } else { // if out of scope, fill with max value
        gs_SortValues[index & 2047] = NULL_KEY;
    }
}

void SaveElement(uint index, uint ListCount) {
    if (index < ListCount)
        g_SortBuffer[index] = gs_SortValues[index & 2047];
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void ComputeMain(uint3 gid: SV_GroupID, uint tid: SV_GroupIndex) {
    // Item index of the start of this group
    const uint offset = gid.x * 2048;
    // Actual number of items that need sorting
#ifdef INDIRECT_DISPATCH
    const uint ListCount = u_CounterBuffer.Load(CounterOffset);
#else
    const uint ListCount = NumElements;
#endif
    
    // Load the data into shared memory
    LoadElement(offset + tid, ListCount);
    LoadElement(offset + tid + 1024, ListCount);
    GroupMemoryBarrierWithGroupSync();
    
    // Sort on shared memory
    uint k;
    // This is better unrolled because it reduces ALU and because some
    // architectures can load/store two LDS items in a single instruction
    // as long as their separation is a compile-time constant.
    [unroll]
    for (k = 2; k <= 2048; k <<= 1) {
        [unroll]
        for (uint j = k / 2; j > 0; j /= 2) {
            uint Index2 = InsertOneBit(tid, j);
            uint Index1 = Index2 ^ (k == 2 * j ? k - 1 : j);

            KEY_TYPE A = gs_SortValues[Index1];
            KEY_TYPE B = gs_SortValues[Index2];

            if (ShouldSwap(A, B)) {
                // Swap the keys
                gs_SortValues[Index1] = B;
                gs_SortValues[Index2] = A;
            }

            GroupMemoryBarrierWithGroupSync();
        }
    }
    
    // Save the data back to global memory
    SaveElement(offset + tid, ListCount);
    SaveElement(offset + tid + 1024, ListCount);
}