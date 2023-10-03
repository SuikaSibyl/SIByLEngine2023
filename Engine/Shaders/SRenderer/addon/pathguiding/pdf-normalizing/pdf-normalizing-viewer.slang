#include "../../../include/common/colormaps.hlsli"

RWTexture2D<float4> u_Output;
RWTexture2D<float4> u_Irradiance;

const Texture2D<float> s_PdfNormalizing;
const Texture2D<uint> s_PdfNormalizingInfo;

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

    const float pdf_unormalized = s_PdfNormalizing[pixel];
    const uint normalizing_factor_u = s_PdfNormalizingInfo[int2(0, 0)];
    const float normalizing_factor = float(normalizing_factor_u) / 1000;
    const double denom = double(normalizing_factor / (resolution.x * resolution.y));
    const double pdf = double(pdf_unormalized) / denom;
    const float val = float(pdf * double(scalar));
    
    const float3 color = viridis(val);

    u_Irradiance[pixel] = float4(float3(pdf), 1.0f);
    u_Output[pixel] = float4(color, 1.0f);
}