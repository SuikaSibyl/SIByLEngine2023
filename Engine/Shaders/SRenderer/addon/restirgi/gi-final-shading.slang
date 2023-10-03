#include "../../include/common/light_impl.hlsli"
#include "../../include/common/packing.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../gbuffer/gbuffer_common.hlsli"
#include "../gbuffer/gbuffer_interface.hlsli"
#include "include/GIReservoir.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    GIResamplingRuntimeParameters runtimeParams;
    uint shadeInputBufferIndex;
    uint giEnableFinalVisibility;
    uint sample_batch;
    bool evaluate_radiance;
    bool extra_bounce;
};

RWStructuredBuffer<PackedGIReservoir> u_GIReservoirs;

RWTexture2D<float4> u_Diffuse;
RWTexture2D<uint> u_Specular;

[shader("raygeneration")]
void RgenMain() {
    // Initialize stage
    // --------------------------------------------------------------------
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = getViewportSize(globalUniform.cameraData);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    RandomSamplerState RNG = InitRandomSampler(pixel, sample_batch, 13);

    // Load Primary Hit from GBuffer
    // --------------------------------------------------------------------
    const Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );

    ShadingSurface primarySurface = GetGBufferSurface(
        pixel, globalUniform.cameraData);

    GIReservoir reservoir = LoadGIReservoir(
        runtimeParams, pixel, shadeInputBufferIndex, u_GIReservoirs);
    
    float3 diffuse = 0;
    float3 specular = 0;

    if (IsShadingSurfaceValid(primarySurface) && IsValidGIReservoir(reservoir)) {
        if (evaluate_radiance) {
            const float3 bsdf_direction = normalize(reservoir.position - primarySurface.worldPos);
            const float bsdf_pdf = PdfBsdfSample(primarySurface, -primaryRay.direction, bsdf_direction);
            const Ray bsdfRay = SpawnRay(primarySurface, bsdf_direction);
            float3 first_bsdf = EvalBsdf(primarySurface, -primaryRay.direction, bsdfRay.direction);
            float3 throughput = float3(1);
            if (bsdf_pdf == 0) throughput = float3(0, 0, 0);
            RNG.state = reservoir.RNGState;
            PrimaryPayload payload;
            payload.RND = RNG;
            float3 radiance = float3(0);
            Intersection(bsdfRay, SceneBVH, payload, RNG);
            if (HasHit(payload.hit)) {
                const PolymorphicLightInfo light = lights[0];
                radiance += EvaluateDirectLight(bsdfRay, payload.hit, light, RNG) * throughput;
                // further bounces
                if (extra_bounce) {
                    radiance += EvaluateMultibounceIndirect(bsdfRay, 1, throughput, payload, RNG);
                }
            }
            reservoir.radiance = radiance;            
            StoreGIReservoir(reservoir, runtimeParams, pixel, shadeInputBufferIndex, u_GIReservoirs);
        }

        float3 radiance = reservoir.radiance * reservoir.weightSum;
        float3 visibility = 1.0;

        if (giEnableFinalVisibility > 0) {
            const Ray ray = SetupVisibilityRay(primarySurface, reservoir.position, 0.01);
            visibility = TraceOccludeRay(ray, RNG, SceneBVH) ? float3(0) : float3(1);
        }
        
        const float3 bsdf_direction = normalize(reservoir.position - primarySurface.worldPos);
        const float3 first_bsdf = EvalBsdf(primarySurface, -primaryRay.direction, bsdf_direction);

        radiance *= visibility * first_bsdf;
        diffuse += radiance;
        // diffuse += visibility * radiance;
    }

    u_Diffuse[pixel] = float4(diffuse, 1);
    u_Specular[pixel] = 0;
}