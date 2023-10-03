#include "tone-mapping.hlsli"
#include "gamma-correction.hlsli"

Texture2D<float4> s_source;

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    int2 resolution;
    float exposure;
    // float3x4 colorTransform;
};

[shader("fragment")]
float4 fragmentMain(in float2 uv: Coord) : SV_Target {
    const int2 pixel = clamp(int2(uv * resolution), 0, resolution - 1);
    float4 color = s_source[pixel].rgba;
    float3 finalColor = color.rgb;
    
    finalColor *= exposure;
    // apply color grading
    // finalColor = mul((float3x3)colorTransform, finalColor);
    // apply tone mapping
    finalColor = ACESToneMapping(finalColor);
    // return final color
    return float4(LinearToScreen(finalColor), color.a);
}