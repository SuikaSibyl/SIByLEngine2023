#include "../../include/common/light_impl.hlsli"
#include "../../include/common/packing.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../gbuffer/gbuffer_common.hlsli"
#include "../gbuffer/gbuffer_interface.hlsli"
#include "include/GIResamplingFunctions.hlsli"
#include "include/GIReservoir.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    GIResamplingRuntimeParameters runtimeParams;
    uint temporalInputBufferIndex;
    uint temporalOutputBufferIndex;
    uint maxHistoryLength;
    uint temporalBiasCorrection;
    float temporalDepthThreshold;
    float temporalNormalThreshold;
    bool enablePermutationSampling;
    bool enableFallbackSampling;
    uint giReservoirMaxAge;
    uint sample_batch;
};

RWStructuredBuffer<PackedGIReservoir> u_GIReservoirs;
RWTexture2D<float4> u_debug;

[shader("raygeneration")]
void RgenMain() {
    // Initialize stage
    // --------------------------------------------------------------------
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = getViewportSize(globalUniform.cameraData);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    RandomSamplerState RNG = InitRandomSampler(pixel, sample_batch, 7);
    
    // Load Primary Hit from GBuffer
    // --------------------------------------------------------------------
    ShadingSurface surface = GetGBufferSurface(
        pixel, globalUniform.cameraData
    );
    surface.bsdfID = 0;

    const float3 motionVector = GetMotionVector(pixel);

    GIReservoir curReservoir = EmptyGIReservoir();
    
    float selectedTargetPdf = 0;

    if (IsShadingSurfaceValid(surface)) {
        GITemporalResamplingParameters tParams;
        tParams.screenSpaceMotion = motionVector;
        tParams.sourceBufferIndex = temporalInputBufferIndex;
        tParams.maxHistoryLength = maxHistoryLength;
        tParams.biasCorrectionMode = temporalBiasCorrection;
        tParams.depthThreshold = temporalDepthThreshold;
        tParams.normalThreshold = temporalNormalThreshold;
        tParams.enablePermutationSampling = enablePermutationSampling;
        tParams.enableFallbackSampling = enableFallbackSampling;

        GIReservoir reservoir = LoadGIReservoir(runtimeParams, pixel, temporalOutputBufferIndex, u_GIReservoirs);

        // Age threshold should vary.
        // This is to avoid to die a bunch of GI reservoirs at once at a disoccluded area.
        tParams.maxReservoirAge = uint(giReservoirMaxAge * (0.5 + GetNextRandom(RNG) * 0.5));

        // Execute resampling.
        curReservoir = GITemporalResampling(
            pixel, surface, reservoir, tParams, runtimeParams,
            prevGlobalUniform.cameraData, u_GIReservoirs, RNG, u_debug);
        // GIReservoir temporalReservoir = LoadGIReservoir(runtimeParams, pixel, temporalInputBufferIndex, u_GIReservoirs);
        // // if (CombineGIReservoirs(reservoir, temporal, GetNextRandom(RNG), target_pdf)) {
        // //     selectedTargetPdf = target_pdf;
        // // }
        // if (IsValidGIReservoir(temporalReservoir)) {
        //     bool foundTemporalReservoir = true;
        //     // Clamp history length
        //     // temporalReservoir.M = min(temporalReservoir.M, tParams.maxHistoryLength);
        //     // // Make the sample older
        //     // ++temporalReservoir.age;
        //     // discard if the reservoir is too old
        //     // if (temporalReservoir.age > tParams.maxReservoirAge)
        //     //     foundTemporalReservoir = false;
        //     if (foundTemporalReservoir) {
        //         float a_selectedTargetPdf = GetGISampleTargetPdfForSurface(temporalReservoir.position, temporalReservoir.radiance, surface);
        //         if (CombineGIReservoirs(curReservoir, temporalReservoir, /* random = */ GetNextRandom(RNG), a_selectedTargetPdf)) {
        //             selectedTargetPdf = a_selectedTargetPdf;
        //         }
        //     }
        // }
    }

    StoreGIReservoir(curReservoir, runtimeParams, pixel, temporalOutputBufferIndex, u_GIReservoirs);
    // StoreGIReservoir(reservoir, runtimeParams, pixel, temporalInputBufferIndex, u_GIReservoirs);

    // // Save the initial sample radiance for MIS in the final shading pass
    // secondaryGBufferData.emission = outputShadingResult ? 0 : radiance;
    // u_SecondaryGBuffer[gbufferIndex] = secondaryGBufferData;
}