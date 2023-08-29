#include "../../include/common/light_impl.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/common/packing.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "gbuffer_common.hlsli"
#include "gbuffer_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    uint sample_batch;
};

RWTexture2D<uint> u_Diffuse;
RWTexture2D<uint> u_Specular;

RWTexture2D<uint> u_RNGSeed;
RWTexture2D<uint> u_RNGPrev;

RWTexture2D<float4> u_Debug;

[shader("raygeneration")]
void RgenMain() {
    // Initialize stage
    // --------------------------------------------------------------------
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = getViewportSize(globalUniform.cameraData);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    RandomSamplerState RNG;
    RNG.state = u_RNGSeed[pixel];
    u_RNGPrev[pixel] = RNG.state;
    
    // Load Primary Hit from GBuffer
    // --------------------------------------------------------------------
    const Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );
    
    ShadingSurface surface = GetGBufferSurface(
        pixel, globalUniform.cameraData
    );

    float3 diffuse = float3(0);
    if (IsShadingSurfaceValid(surface)) {
        const PolymorphicLightInfo light = lights[0];
        const float3 di = EvaluateDirectLight(primaryRay, surface, light, RNG);

        float bsdf_pdf;
        Ray bsdfRay = SpawnBsdfRay(surface, -primaryRay.direction, RNG, bsdf_pdf);
        float3 first_bsdf = EvalBsdf(surface, -primaryRay.direction, bsdfRay.direction);
        float3 demodulate = (sign(surface.diffuseAlbedo) == float3(1)) ? float3(1) / surface.diffuseAlbedo : float3(0);
        float3 throughput = first_bsdf / bsdf_pdf * demodulate;
        if (bsdf_pdf == 0) throughput = float3(0, 0, 0);

        PrimaryPayload payload;
        Intersection(bsdfRay, SceneBVH, payload, RNG);
        if (HasHit(payload.hit)) {
            const PolymorphicLightInfo light = lights[0];
            diffuse += EvaluateDirectLight(bsdfRay, payload.hit, light, RNG) * throughput;
        }

    }

    u_Debug[pixel] = float4(diffuse, 1);
    u_RNGSeed[pixel] = RNG.state;
    u_Diffuse[pixel] = PackRGBE(diffuse);
    u_Specular[pixel] = PackRGBE(float3(0));
}