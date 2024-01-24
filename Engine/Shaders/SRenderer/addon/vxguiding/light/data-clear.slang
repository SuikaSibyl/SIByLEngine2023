RWStructuredBuffer<uint> u_hashRecord;
RWStructuredBuffer<uint2> u_irradiance;
RWStructuredBuffer<uint4> u_boundMin;
RWStructuredBuffer<uint4> u_boundMax;

[[vk::push_constant]]
cbuffer PushConstants {
    uint cell_count;
};

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(uint3 DTid: SV_DispatchThreadID) {
    if (DTid.x >= cell_count) return;
    u_hashRecord[DTid.x] = 0xffffffffu;
    u_irradiance[DTid.x] = uint2(0);
    u_boundMin[DTid.x] = uint4(uint3(0xffffffff), 0);
    u_boundMax[DTid.x] = uint4(uint3(0), 0);
}