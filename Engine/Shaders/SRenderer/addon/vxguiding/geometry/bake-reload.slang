#include "common.hlsli"

StructuredBuffer<uint4> u_bakedMin;
StructuredBuffer<uint4> u_bakedMax;
StructuredBuffer<uint> u_bakedHashRecord;
StructuredBuffer<uint2> u_irradiance;

RWStructuredBuffer<uint4> u_boundMin;
RWStructuredBuffer<uint4> u_boundMax;
RWStructuredBuffer<uint> u_hashRecord;

[[vk::push_constant]]
cbuffer PushConstants {
    uint cell_count;
};

groupshared uint gs_hashRecords[512];

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(uint3 DTid: SV_DispatchThreadID) {
    if (DTid.x >= cell_count) return;

    if (u_config.mode == 0) {
        const uint gs_id = DTid.x % 512;
        gs_hashRecords[gs_id] = u_hashRecord[DTid.x];
        const uint baked_rec = u_bakedHashRecord[DTid.x];

        GroupMemoryBarrier();
        
        if (u_irradiance[DTid.x].x != 0) return;
        
        const uint32_t bsize = u_config.bucket_size;
        const uint offset = (DTid.x / bsize) * bsize;
        const uint base = (offset / 512) * 512;
        for (uint i = offset; i < offset + bsize; ++i) {
            if (gs_hashRecords[i - base] == baked_rec) {
                u_boundMin[i] = u_bakedMin[DTid.x];
                u_boundMax[i] = u_bakedMax[DTid.x];
                break;
            }
        }
    } else {
        if (u_irradiance[DTid.x].x != 0) {
            u_hashRecord[DTid.x] = u_bakedHashRecord[DTid.x];
            u_boundMin[DTid.x] = u_bakedMin[DTid.x];
            u_boundMax[DTid.x] = u_bakedMax[DTid.x];
        }
    }
}