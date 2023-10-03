#include "include/gaussian_mm.hlsli"
#include "pdf-accumulation/pdf-accumulator.hlsli"
#include "../../include/common/random.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    int2 debug_pixel;
    uint rnd_seed;
};

Texture2D<float4> u_gmmStatisticsPack0;
Texture2D<float4> u_gmmStatisticsPack1;

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    const int2 pixel = dtid.xy;
    if (any(pixel >= resolution)) return;

    RandomSamplerState RNG = InitRandomSampler(pixel, rnd_seed);

    const float4 pack0 = u_gmmStatisticsPack0[debug_pixel];
    const float4 pack1 = u_gmmStatisticsPack1[debug_pixel];
    GMMStatictics GMMstat = UnpackStatistics(pack0, pack1);
    MultivariateGaussian2D GMM = CreateDistribution(GMMstat);
    
    float2 sample = GMM.DrawSample(float2(GetNextRandom(RNG), GetNextRandom(RNG)));
    AddOneSample2Info();
    AddOneSample2Map(int2(sample * resolution), resolution);

    // const uint sample_count = s_PdfAccumulator[pixel];
    // const uint sample_count_all = s_PdfAccumulatorInfo[int2(0, 0)];
    // const double denom = double(sample_count_all / (resolution.x * resolution.y));
    // const double pdf = double(sample_count) / denom;
    // const float val = float(pdf * double(scalar));

    // const float3 color = viridis(val);

    // u_Output[pixel] = float4(color, 1.0f);
}