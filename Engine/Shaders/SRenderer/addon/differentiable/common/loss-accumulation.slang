const Texture2D<float> u_error;
RWByteAddressBuffer u_error_sum;

[[vk::push_constant]]
cbuffer PushConstants {
    int2 image_size;
};

groupshared float u_error_shared[8];

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int gid: SV_GroupIndex,
) {
    const int2 pixel = dtid.xy;
    const int wid = gid / WaveGetLaneCount();
    
    // Accumulate error within wave
    float error = 0.f;
    if (all(pixel < image_size))
        error = u_error[pixel];
    const float error_wave = WaveActiveSum(error);
    
    // Accumulate error within group
    if (WaveIsFirstLane())
        u_error_shared[wid] = error_wave;
    GroupMemoryBarrierWithGroupSync();
    float error_group = 0.f;
    if (gid < 8) {
        error_group = u_error_shared[gid];
    }
    if (wid == 0) {
        const float error_sum = WaveActiveSum(error_group);
        if (WaveIsFirstLane())
            u_error_sum.InterlockedAddF32(0, error_sum);
    }
}