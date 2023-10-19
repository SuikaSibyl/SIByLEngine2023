#include "../../raytracer/spt_interface.hlsli"
#include "../../include/common/light_impl.hlsli"
#include "../../include/common/sampling.hlsli"

Sampler2D<float4> u_target;
RWTexture2D<float4> u_color;
RWTexture2D<float4> u_output;

[[vk::push_constant]]
cbuffer PushConstants {
    float learning_rate;
    bool learn;
    bool init;
    uint rand_seed;
};

[Differentiable]
float myFunc(float3 albedo, no_diff SplitShading split, no_diff float3 target) {
    return distance(albedo * split.diffuse, target);
}

[shader("raygeneration")]
void RgenMain() {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(DispatchRaysDimensions().xy);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    RandomSamplerState RNG = InitRandomSampler(pixel, rand_seed);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;
    const float2 jitter_pixel = pixel + GetNextRandomFloat2(RNG) - float2(0.5);
    const Ray primaryRay = generateRay(jitter_pixel, resolution, false, globalUniform.cameraData);

    PrimaryPayload payload;
    payload.RND = RNG;
    Intersection(primaryRay, SceneBVH, payload, RNG);
    // If we hit nothing, just output the background color:
    if (init) {
        if (all(pixel < int2(512))) {
            u_color[pixel] = float4(0.5, 0.5, 0.5, 1);
        }
    }
    if (!HasHit(payload.hit)) {
        u_output[pixel] = float4(float3(0), 1);
        return;
    }

    const PolymorphicLightInfo light = lights[0];
    SplitShading split = EvaluateDirectLightSplit(primaryRay, payload.hit, light, RNG);

    const float2 uv = payload.hit.texcoord;
    const int2 texel = int2(uv * 512);
    float3 albedo = u_color[texel].xyz;
    float3 target = u_target.Sample((jitter_pixel + 0.5) / resolution).xyz;
    
    u_output[pixel] = float4(split.diffuse * albedo, 1);
    if (learn) {
        var da = diffPair(albedo);
        bwd_diff(myFunc)(da, split, target, 1.0);
        
        float3 update = albedo - learning_rate * da.d;
        u_color[texel] = float4(update, 1);
    }
}