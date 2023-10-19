#include "../../raytracer/spt_interface.hlsli"
#include "../../include/common/light_impl.hlsli"
#include "../../include/common/sampling.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    uint rand_seed;
};

RWTexture2D<float4> u_output;

[shader("raygeneration")]
void RgenMain() {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(DispatchRaysDimensions().xy);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;
    RandomSamplerState RNG = InitRandomSampler(pixel, rand_seed);
    const float2 jitter_pixel = pixel + GetNextRandomFloat2(RNG) - float2(0.5);
    const Ray primaryRay = generateRay(jitter_pixel, resolution, false, globalUniform.cameraData);
    
    PrimaryPayload payload;
    payload.RND = RNG;
    Intersection(primaryRay, SceneBVH, payload, RNG);

    // If we hit nothing, just output the background color:
    if (!HasHit(payload.hit)) {
        u_output[pixel] = float4(float3(0), 1);
        return;
    }

    float3 radiance = float3(0);
    const PolymorphicLightInfo light = lights[0];
    radiance += EvaluateDirectLight(primaryRay, payload.hit, light, RNG);
    u_output[pixel] = float4(radiance, 1);
}