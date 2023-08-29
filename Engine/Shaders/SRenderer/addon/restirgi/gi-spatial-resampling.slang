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
    uint spatialInputBufferIndex;
    uint spatialOutputBufferIndex;
    uint sample_batch;
    float spatialDepthThreshold;
    float spatialNormalThreshold;
    uint numSpatialSamples;
    float spatialSamplingRadius;
};

RWStructuredBuffer<PackedGIReservoir> u_GIReservoirs;
const StructuredBuffer<uint8_t> u_NeighborOffsets;
const RWTexture2D<float4> u_debug;

[shader("raygeneration")]
void RgenMain() {
    // Initialize stage
    // --------------------------------------------------------------------
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = getViewportSize(globalUniform.cameraData);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    RandomSamplerState RNG = InitRandomSampler(pixel, sample_batch, 8);
    
    // Load Primary Hit from GBuffer
    // --------------------------------------------------------------------
    ShadingSurface surface = GetGBufferSurface(pixel, globalUniform.cameraData);
    
    GIReservoir reservoir = LoadGIReservoir(runtimeParams, pixel, spatialInputBufferIndex, u_GIReservoirs);

    if (IsShadingSurfaceValid(surface)) {
        GISpatialResamplingParameters sparams;
        sparams.sourceBufferIndex = spatialInputBufferIndex;
        sparams.biasCorrectionMode = 2;
        sparams.depthThreshold = spatialDepthThreshold;
        sparams.normalThreshold = spatialNormalThreshold;
        sparams.numSamples = numSpatialSamples;
        sparams.samplingRadius = spatialSamplingRadius;
        
        // Execute resampling.
        reservoir = GISpatialResampling(
            pixel, surface, reservoir, sparams, runtimeParams,
            globalUniform.cameraData, RNG, u_GIReservoirs, u_NeighborOffsets, u_debug);
    }

    StoreGIReservoir(reservoir, runtimeParams, pixel, spatialOutputBufferIndex, u_GIReservoirs);
}