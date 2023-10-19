#include "../../include/common/light_impl.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/common/vmf.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/primitives/quad.hlsli"
#include "../../raytracer/primitives/sphere.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../gbuffer/gbuffer_interface.hlsli"
#include "../gbuffer/gbuffer_prev_interface.hlsli"
#include "../restirgi/include/GIResamplingFunctions.hlsli"
#include "../restirgi/include/GIReservoir.hlsli "
#include "include/pathguiding.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        GIResamplingRuntimeParameters runtimeParams;
        uint2 resolution;
        uint sample_batch;
        uint strategy;
        bool adaption;
        float exponential_factor;
        bool extra_bounce;
        int spp;
        bool extra_half_spp;
        bool learn_first;
        uint initialOutputBufferIndex;
    } pConst;
};

const Texture2D<float4> u_vMFStatisticsPrev;
const Texture2D<uint> u_epochCounterPrev;

RWTexture2D<uint> u_epochCounter;
RWTexture2D<float4> u_vpl0;
RWTexture2D<float4> u_vpl1;
RWTexture2D<float4> u_color;
RWTexture2D<float4> u_vMFStatistics;
RWStructuredBuffer<PackedGIReservoir> u_GIReservoirs;

[shader("raygeneration")]
void RgenMain() {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(pConst.resolution);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;
    RandomSamplerState RNG = InitRandomSampler(pixel, pConst.sample_batch);
    // Reservoir for ReSTIR
    GIReservoir reservoir = EmptyGIReservoir();
    float selectedTargetPdf = 0.0f;

    // Load Primary Hit from VBuffer
    // --------------------------------------------------------------------
    const Ray primaryRay = generateRay(pixel, resolution, false, globalUniform.cameraData);
    const ShadingSurface surface = GetGBufferSurface(pixel, globalUniform.cameraData);
    // Load the previous GBuffer surface
    const float3 motionVector = GetMotionVector(pixel);
    const int2 pixel_prev = int2(round(float2(pixel) + motionVector.xy));
    const ShadingSurface temporalSurface = GetPrevGBufferSurface(pixel_prev, prevGlobalUniform.cameraData);
    const float expectedPrevLinearDepth = surface.viewDepth + motionVector.z;

    // If the surface is invalid, we're done
    // --------------------------------------------------------------------
    if (!IsShadingSurfaceValid(surface)) {
        // save the vMF statistics
        u_vMFStatistics[pixel] = float4(float3(0), 1);
        u_epochCounter[pixel] = 0;
        u_vpl0[pixel] = float4(0);
        u_vpl1[pixel] = float4(0);
        // save the reservoir
        StoreGIReservoir(reservoir, pConst.runtimeParams, pixel, 
            pConst.initialOutputBufferIndex, u_GIReservoirs);
        return;
    }

    // Load the vMF statistics
    // --------------------------------------------------------------------
    vMFMLEStatistics vMFStat = vMFMLEStatistics(float4(0));
    int epochCounter = 0;
    { // constant params
        const float depthThreshold = 0.1;
        const float normalThreshold = 0.5;
        // skip the sample if the surface is invalid
        if (IsShadingSurfaceValid(temporalSurface) &&
            IsValidNeighbor(surface.geometryNormal, temporalSurface.geometryNormal,
                            expectedPrevLinearDepth, temporalSurface.viewDepth,
                            normalThreshold, depthThreshold) &&
            AreMaterialsSimilar(surface, temporalSurface)
        ) {
            vMFStat = vMFMLEStatistics(u_vMFStatisticsPrev[pixel_prev]);
            epochCounter = u_epochCounterPrev[pixel_prev];
        }
    }
    vMFDistribution vMF = vMFDistribution(vMFStat);
    float3 radiance = float3(0);

    // Begin the sampling
    // --------------------------------------------------------------------
    PrimaryPayload payload; payload.RND = RNG;
    { // First draw BSDF samples
        float bsdf_pdf;
        Ray bsdfRay = SpawnBsdfRay(surface, -primaryRay.direction, RNG, bsdf_pdf);
        const uint32_t RNGState = RNG.state;
        float3 di = float3(0); float3 throughput = float3(1);
        float3 indirect = EvaluateIndirectIllumination(
            primaryRay, bsdfRay,
            double(bsdf_pdf),
            surface, payload, RNG, di, throughput);
        const float3 vpl_position = payload.hit.position;
        if (pConst.extra_bounce) {
            indirect += EvaluateMultibounceIndirect(bsdfRay, 1, throughput, payload, RNG);
        }

        if (any(indirect > 0)) {
            float w1 = bsdf_pdf;
            float w2 = vMF.Pdf(bsdfRay.direction);
            // power heuristic
            w1 = w1 * w1;
            w2 = w2 * w2;
            float mis_weight = w1 / (w1 + w2);
            radiance += indirect * mis_weight;
            // Add the first sample to the reservoir
            GIReservoir local = MakeGIReservoir(payload.hit.position, payload.hit.geometryNormal, di, bsdf_pdf, RNGState);
            const float target_pdf = GetGISampleTargetPdfForSurface(local.position, local.radiance, surface, globalUniform.cameraData.posW);
            if (CombineGIReservoirs(reservoir, local, GetNextRandom(RNG), target_pdf * mis_weight)) {
                selectedTargetPdf = target_pdf;
            }
        }
        // Save the first sample to the VPL buffer
        const float sumweight = (bsdf_pdf != 0) ? luminance(di) / bsdf_pdf : 0;
        u_vpl0[pixel] = float4(vpl_position, sumweight);
    }
    { // Second draw SSPG samples
        const float3 guidedDir = vMF.DrawSample(float2(GetNextRandom(RNG), GetNextRandom(RNG)));
        const float guidedPdf = vMF.Pdf(guidedDir);
        Ray guideRay = SpawnRay(surface, guidedDir);
        float3 di = float3(0); float3 throughput = float3(1);
        float3 indirect = EvaluateIndirectIllumination(
            primaryRay, guideRay,
            double(guidedPdf),
            surface, payload, RNG, di, throughput);
        const float3 vpl_position = payload.hit.position;
        if (pConst.extra_bounce) {
            indirect += EvaluateMultibounceIndirect(guideRay, 1, throughput, payload, RNG);
        }

        if (any(indirect > 0)) {
            float w1 = PdfBsdfSample(surface, -primaryRay.direction, guidedDir);
            float w2 = guidedPdf;
            w1 = w1 * w1;
            w2 = w2 * w2;
            if (isnan(w1)) w1 = 0;
            float mis_weight = w2 / (w1 + w2);
            radiance += indirect * mis_weight;
            // Add the second sample to the reservoir
            GIReservoir local = MakeGIReservoir(payload.hit.position, payload.hit.geometryNormal, di, guidedPdf, 0);
            const float target_pdf = GetGISampleTargetPdfForSurface(local.position, local.radiance, surface, globalUniform.cameraData.posW);
            if (CombineGIReservoirs(reservoir, local, GetNextRandom(RNG), target_pdf * mis_weight)) {
                selectedTargetPdf = target_pdf;
            }
        }
        // Save the second sample to the VPL buffer
        const float sumweight = (guidedPdf != 0) ? luminance(di) / guidedPdf : 0;
        u_vpl1[pixel] = float4(vpl_position, sumweight);
    }

    u_vMFStatistics[pixel] = vMFStat.Pack();
    u_epochCounter[pixel] = epochCounter;
    u_color[pixel] = float4(radiance, 1);
    
    // Finalize the sampling for ReSTIR
    reservoir.M = 1;
    const float normalizationNumerator = 1.0;
    const float normalizationDenominator = selectedTargetPdf;
    FinalizeGIResampling(reservoir, normalizationNumerator, normalizationDenominator);
    StoreGIReservoir(reservoir, pConst.runtimeParams, pixel, pConst.initialOutputBufferIndex, u_GIReservoirs);
}