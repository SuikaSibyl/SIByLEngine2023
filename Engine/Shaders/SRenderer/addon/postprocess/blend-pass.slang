#include "tone-mapping.hlsli"
#include "gamma-correction.hlsli"

Texture2D<float4> u_input;
Texture2D<float4> u_blend;

enum class BlendMode {
    Normal,
    Dissolve,
    Darken,
    Multiply,
};

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    int2 resolution;
    BlendMode blendMode;
    uint32_t rand_seed;
    float exposure;
};

[shader("fragment")]
float4 fragmentMain(in float2 uv: Coord) : SV_Target {
    const int2 pixel = clamp(int2(uv * resolution), 0, resolution - 1);
    const float3 color = u_input[pixel].rgb;
    const float3 blend = u_blend[pixel].rgb;

    float3 finalColor = float3(1, 0, 1);
    switch (blendMode) {
    case BlendMode::Normal:
        finalColor = blend; break;
    case BlendMode::Dissolve:
        finalColor = blend; break; // TODO
    case BlendMode::Darken:
        finalColor = blend; break; // TODO
    case BlendMode::Multiply:
        finalColor = color * blend; break;
    }
    
    return float4(finalColor, 1);
}