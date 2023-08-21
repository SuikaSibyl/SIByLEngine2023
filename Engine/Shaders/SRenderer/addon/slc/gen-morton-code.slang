#include "../../include/common/space_filling_curve.hlsli"

StructuredBuffer<float4> u_vplPositions;
RWStructuredBuffer<uint64_t> u_keyIndexList;
ByteAddressBuffer u_vplCounter;

[[vk::push_constant]]
cbuffer PushConstants {
    float3 minExtent;
    int padding;
    float3 maxExtent;
};

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(uint3 DTid : SV_DispatchThreadID) {
	const uint numVpls = u_vplCounter.Load(0);
	if (DTid.x < numVpls) {
        // normalize position to [0,1]
        const float3 position = u_vplPositions[DTid.x].xyz;
        const float3 unipos = (position - minExtent) / (maxExtent - minExtent);
        const uint64_t posCode = ZCurve3DToMortonCode(unipos);
        const uint64_t idCode = DTid.x;
        const uint64_t LongKey = (posCode << 32) | (idCode << 0);
        u_keyIndexList[DTid.x] = LongKey;
	}
}