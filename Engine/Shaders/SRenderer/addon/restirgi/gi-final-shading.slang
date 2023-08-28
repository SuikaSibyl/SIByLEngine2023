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
};

const StructuredBuffer<PackedGIReservoir> u_GIReservoirs;

RWTexture2D<float4> u_Diffuse;
RWTexture2D<uint> u_Specular;
// float GetMISWeight(const SplitBrdf roughBrdf, const SplitBrdf trueBrdf, const float3 diffuseAlbedo) {
//     float3 combinedRoughBrdf = roughBrdf.demodulatedDiffuse * diffuseAlbedo + roughBrdf.specular;
//     float3 combinedTrueBrdf = trueBrdf.demodulatedDiffuse * diffuseAlbedo + trueBrdf.specular;

//     combinedRoughBrdf = clamp(combinedRoughBrdf, 1e-4, kMaxBrdfValue);
//     combinedTrueBrdf = clamp(combinedTrueBrdf, 0, kMaxBrdfValue);

//     const float initWeight = saturate(calcLuminance(combinedTrueBrdf) / calcLuminance(combinedTrueBrdf + combinedRoughBrdf));
//     return initWeight * initWeight * initWeight;
// }

// GIReservoir LoadInitialSampleReservoir(int2 reservoirPosition, RAB_Surface primarySurface) {
//     const uint gbufferIndex = RTXDI_ReservoirPositionToPointer(g_Const.runtimeParams, reservoirPosition, 0);
//     const SecondaryGBufferData secondaryGBufferData = u_SecondaryGBuffer[gbufferIndex];

//     const float3 normal = octToNdirUnorm32(secondaryGBufferData.normal);
//     const float3 throughput = Unpack_R16G16B16A16_FLOAT(secondaryGBufferData.throughputAndFlags).rgb;

//     // Note: the secondaryGBufferData.emission field contains the sampled radiance saved in ShadeSecondarySurfaces
//     return RTXDI_MakeGIReservoir(secondaryGBufferData.worldPos,
//                                  normal, secondaryGBufferData.emission * throughput, secondaryGBufferData.pdf);
// }

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

    const ShadingSurface primarySurface = GetGBufferSurface(
        pixel, globalUniform.cameraData);

    const GIReservoir reservoir = LoadGIReservoir(
        runtimeParams, pixel, shadeInputBufferIndex, u_GIReservoirs);
    
    float3 diffuse = 0;
    float3 specular = 0;

    if (IsShadingSurfaceValid(primarySurface) && IsValidGIReservoir(reservoir)) {
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