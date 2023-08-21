/***************************************************************************
 # Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#include "include/GIResamplingFunctions.hlsli"

[shader("raygeneration")]
void RgenMain() {
    uint2 GlobalIndex = DispatchRaysIndex().xy;
    uint2 pixelPosition = RTXDI_ReservoirPosToPixelPos(GlobalIndex, g_Const.runtimeParams);
    
    RAB_RandomSamplerState rng = RAB_InitRandomSampler(GlobalIndex, 7);
    
    const RAB_Surface primarySurface = RAB_GetGBufferSurface(pixelPosition, false);
    
    const uint2 reservoirPosition = RTXDI_PixelPosToReservoirPos(pixelPosition, g_Const.runtimeParams);
    RTXDI_GIReservoir reservoir = RTXDI_LoadGIReservoir(g_Const.runtimeParams, reservoirPosition, g_Const.initialOutputBufferIndex);

    float3 motionVector = t_MotionVectors[pixelPosition].xyz;
    motionVector = convertMotionVectorToPixelSpace(g_Const.view, g_Const.prevView, pixelPosition, motionVector);

    if (RAB_IsSurfaceValid(primarySurface)) {
        RTXDI_GISpatioTemporalResamplingParameters stParams;

        stParams.screenSpaceMotion = motionVector;
        stParams.sourceBufferIndex = g_Const.temporalInputBufferIndex;
        stParams.maxHistoryLength = g_Const.maxHistoryLength;
        stParams.biasCorrectionMode = g_Const.temporalBiasCorrection;
        stParams.depthThreshold = g_Const.temporalDepthThreshold;
        stParams.normalThreshold = g_Const.temporalNormalThreshold; 
        stParams.enablePermutationSampling = g_Const.enablePermutationSampling;
        stParams.numSpatialSamples = g_Const.numSpatialSamples;
        stParams.samplingRadius = g_Const.spatialSamplingRadius;
        stParams.enableFallbackSampling = g_Const.enableFallbackSampling;

        // Age threshold should vary.
        // This is to avoid to die a bunch of GI reservoirs at once at a disoccluded area.
        stParams.maxReservoirAge = g_Const.giReservoirMaxAge * (0.5 + RAB_GetNextRandom(rng) * 0.5);

        // Execute resampling.
        reservoir = RTXDI_GISpatioTemporalResampling(pixelPosition, primarySurface, reservoir, rng, stParams, g_Const.runtimeParams);
    }
    
    RTXDI_StoreGIReservoir(reservoir, g_Const.runtimeParams, reservoirPosition, g_Const.spatialOutputBufferIndex);
}
