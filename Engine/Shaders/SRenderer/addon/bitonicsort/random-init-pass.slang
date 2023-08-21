#include "../../include/common/cpp_compatible.hlsli"
#include "../../include/common/random.hlsli"
#include "bitonic_common.hlsli"

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    struct PushConstants {
        uint numElements;
        uint randomSeed;
    } pConst;
};

RWStructuredBuffer<KEY_TYPE> g_SortBuffer;

KEY_TYPE RandomKey(inout_ref(RandomSamplerState) RNG) {
#if KEY_TYPE_ENUM == 0
    return GetNextRandomUint(RNG);
#elif KEY_TYPE_ENUM == 1
    uint64_t key = GetNextRandomUint(RNG);
    key <<= 32;
    key |= GetNextRandomUint(RNG);
    return key;
#endif
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void ComputeMain(uint3 gid: SV_GroupID, uint tid: SV_GroupIndex) {
    // Item index of the start of this group
    const uint offset = gid.x * 2048;
    // Actual number of items that need sorting
    RandomSamplerState RNG = InitRandomSampler(uint2(gid.x, tid), pConst.randomSeed);
    g_SortBuffer[offset + tid] = RandomKey(RNG);
    g_SortBuffer[offset + tid + 1024] = RandomKey(RNG);
}