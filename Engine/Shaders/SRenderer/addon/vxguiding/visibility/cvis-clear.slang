RWTexture2D<uint> u_spixel_visibility; // 40 * 23
RWTexture2D<uint> u_spixel_counter;    // 40 * 23
RWStructuredBuffer<uint> u_cluster_counter;      // 32

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
};

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
) {
    if (any(DTid.xy >= resolution)) return;
    u_spixel_visibility[DTid.xy] = 0;
    u_spixel_counter[DTid.xy] = 0;
    
    if(DTid.x < 32 && DTid.y == 0) {
        u_cluster_counter[DTid.x] = 0;
    }
}