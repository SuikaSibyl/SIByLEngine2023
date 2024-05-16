#include "common.hlsli"

RWStructuredBuffer<float> u_image;
RWStructuredBuffer<float3> u_trans;

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(512, 512);
    const int2 pixel = int2(dtid.xy);

    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;
    
    Ray ray = generateRay(pixel, resolution);
    RayDifferential raydiff = generateRayDifferential(pixel, resolution);

    const float3 translation = u_trans[0]; // gt: float3(0.1, -0.1, 0.1)
    HitAD reference = ray_tracing(ray, translation);
    u_image[pixel.x * 512 + pixel.y] = reference.color.x;
}