#include "common/colormaps.hlsli"

[[vk::push_constant]] cbuffer PushConstants {
    float vmax;
    int colormap;
};

Texture2D<float> pmf_ideal_texture;
Texture2D<float> pmf_actual_texture;
StructuredBuffer<float> pmf_ideal_summation;
StructuredBuffer<float> pmf_actual_summation;

RWTexture2D<float4> pmf_ideal_visualize;
RWTexture2D<float4> pmf_actual_visualize;

float3 colormapping(float v) {
    v *=  512 * 512 / vmax;
    if (colormap == 0) {
        return viridis(v);
    } else if (colormap == 1) {
        return magma(v);
    } else if (colormap == 2) {
        return inferno(v);
    } else if (colormap == 3) {
        return plasma(v);
    } else {
        return float3(v);
    }
}

[shader("compute")]
[numthreads(32, 32, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    const int2 texel = int2(dtid.xy);
    const float v_ideal = pmf_ideal_texture[texel].x / pmf_ideal_summation[0];
    const float v_actual = pmf_actual_texture[texel].x / pmf_actual_summation[0];
    
    pmf_ideal_visualize[texel] = float4(colormapping(v_ideal), 1.0f);
    pmf_actual_visualize[texel] = float4(colormapping(v_actual), 1.0f);
}