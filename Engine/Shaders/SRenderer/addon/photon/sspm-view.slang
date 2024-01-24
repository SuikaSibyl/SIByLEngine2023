#include "common.hlsli"

RWByteAddressBuffer u_photon_map;
RWTexture2D<float4> u_color;

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
};

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(uint3 DTid: SV_DispatchThreadID) {
    const int2 pixel = int2(DTid.xy);
    if (any(pixel >= resolution)) return;
    // Clear the screen space photon map
    const uint flatten = FlattensPixelToIndex(pixel, resolution.x);
    const float3 color = u_photon_map.Load<float3>(flatten * sizeof(float3));
    u_color[pixel] = float4(color, 1.f);
}