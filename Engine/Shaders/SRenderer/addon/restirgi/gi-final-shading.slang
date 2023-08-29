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
            PrimaryPayload payload;
            Intersection(bsdfRay, SceneBVH, payload, RNG);
            if (HasHit(payload.hit)) {
                const PolymorphicLightInfo light = lights[0];
                reservoir.radiance = (bsdf_pdf != 0)
                                      ? EvaluateDirectLight(bsdfRay, payload.hit, light, RNG)
                                      : float3(0);
            }
            else {
                reservoir.radiance = float3(0);
            }
            StoreGIReservoir(reservoir, runtimeParams, pixel, shadeInputBufferIndex, u_GIReservoirs);
        }

        float3 radiance = reservoir.radiance * reservoir.weightSum;
        float3 visibility = 1.0;

        if (giEnableFinalVisibility > 0) {
            const Ray ray = SetupVisibilityRay(primarySurface, reservoir.position, 0.01);
            visibility = TraceOccludeRay(ray, RNG, SceneBVH) ? float3(0) : float3(1);
            if (dot(primarySurface.geometryNormal, ray.direction) < 0) visibility = float3(0);
        }
        
        const float3 bsdf_direction = normalize(reservoir.position - primarySurface.worldPos);
        const float3 first_bsdf = EvalBsdf(primarySurface, -primaryRay.direction, bsdf_direction);

        radiance *= visibility * first_bsdf;
        diffuse += radiance;
        // const SplitBrdf brdf = EvaluateBrdf(primarySurface, reservoir.position);

        // if (g_Const.giEnableFinalMIS) {
        //     const RTXDI_GIReservoir initialReservoir = LoadInitialSampleReservoir(reservoirPosition, primarySurface);
        //     const SplitBrdf brdf0 = EvaluateBrdf(primarySurface, initialReservoir.position);

        //     RAB_Surface roughenedSurface = primarySurface;
        //     roughenedSurface.roughness = max(roughenedSurface.roughness, kMISRoughness);

        //     const SplitBrdf roughBrdf = EvaluateBrdf(roughenedSurface, reservoir.position);
        //     const SplitBrdf roughBrdf0 = EvaluateBrdf(roughenedSurface, initialReservoir.position);

        //     const float finalWeight = 1.0 - GetMISWeight(roughBrdf, brdf, primarySurface.diffuseAlbedo);
        //     const float initialWeight = GetMISWeight(roughBrdf0, brdf0, primarySurface.diffuseAlbedo);

        //     const float3 initialRadiance = initialReservoir.radiance * initialReservoir.weightSum;

        //     diffuse = brdf.demodulatedDiffuse * radiance * finalWeight
        //             + brdf0.demodulatedDiffuse * initialRadiance * initialWeight;

        //     specular = brdf.specular * radiance * finalWeight
        //              + brdf0.specular * initialRadiance * initialWeight;
        // }
        // else {
        //     diffuse = brdf.demodulatedDiffuse * radiance;
        //     specular = brdf.specular * radiance;
        // }

        // specular = DemodulateSpecular(primarySurface.specularF0, specular);
    }

    u_Diffuse[pixel] = float4(diffuse, 1);
    u_Specular[pixel] = 0;
    // StoreShadingOutput(GlobalIndex, pixelPosition,
    //                    primarySurface.viewDepth, primarySurface.roughness, diffuse, specular, 0, false, true);
}