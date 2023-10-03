#include "../../../include/common/colormaps.hlsli"

RWTexture2D<float4> u_Output;
RWTexture2D<float4> u_Importance;

const Texture2D<uint> s_PdfAccumulator;
const Texture2D<uint> s_PdfAccumulatorInfo;

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    float scalar;
};

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    const int2 pixel = dtid.xy;
    if (any(pixel >= resolution)) return;
    
    const uint sample_count = s_PdfAccumulator[pixel];
    const uint sample_count_all = s_PdfAccumulatorInfo[int2(0, 0)];
    const double denom = double(sample_count_all / (resolution.x * resolution.y));
    const double pdf = double(sample_count) / denom;
    const float val = float(pdf * double(scalar));

    const float3 color = viridis(val);

    u_Output[pixel] = float4(color, 1.0f);
    u_Importance[pixel] = float4(float3(pdf), 1.0f);
}