RWByteAddressBuffer u_IndirectArgsBuffer;
ByteAddressBuffer u_CounterBuffer;

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    uint MaxIterations;
    uint CounterOffset;
};

uint NextPow2(uint Val) {
    uint Mask = (1 << firstbithigh(Val)) - 1;
    return (Val + Mask) & ~Mask;
}

[shader("compute")]
[numthreads(22, 1, 1)]
void ComputeMain(uint GI: SV_GroupIndex) {
    if (GI >= MaxIterations)
        return;
    uint ListCount = u_CounterBuffer.Load(int(CounterOffset));
    uint k = 2048 << GI;
    // We need one more iteration every time the number of thread groups doubles
    if (k > NextPow2((ListCount + 2047) & ~2047))
        ListCount = 0;
    uint PrevDispatches = GI * (GI + 1) / 2;
    uint Offset = 12 * PrevDispatches;
    // Generate outer sort dispatch arguments
    for (uint j = k / 2; j > 1024; j /= 2) {
        // All of the groups of size 2j that are full
        uint CompleteGroups = (ListCount & ~(2 * j - 1)) / 2048;
        // Remaining items must only be sorted if there are more than j of them
        uint PartialGroups = ((uint)max(int(ListCount - CompleteGroups * 2048 - j), 0) + 1023) / 1024;
        u_IndirectArgsBuffer.Store3(Offset, uint3(CompleteGroups + PartialGroups, 1, 1));
        Offset += 12;
    }
    // The inner sort always sorts all groups (rounded up to multiples of 2048)
    u_IndirectArgsBuffer.Store3(Offset, uint3((ListCount + 2047) / 2048, 1, 1));
}
