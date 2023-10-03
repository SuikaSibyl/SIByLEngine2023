const Texture2D<float> s_PdfNormalizing;
RWTexture2D<uint> u_PdfNormalizingInfo;

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
};

groupshared uint radiance_group;

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int gid: SV_GroupIndex,
) {
    const int2 pixel = dtid.xy;
    if (gid == 0) radiance_group = 0u;
    GroupMemoryBarrierWithGroupSync();

    float radiance = 0.f;
    // If the pixel is in the range of the image, 
    // then we can read from the texture
    if (all(pixel < resolution)) {
        radiance = s_PdfNormalizing[pixel];
    }

    // Accumulate the radiance
    float radiance_wave = WaveActiveSum(radiance);
    if (WaveIsFirstLane()) {
        InterlockedAdd(radiance_group, uint(radiance_wave * 1000));
    }
    GroupMemoryBarrierWithGroupSync();

    if (gid == 0)
        InterlockedAdd(u_PdfNormalizingInfo[int2(0)], radiance_group);
}