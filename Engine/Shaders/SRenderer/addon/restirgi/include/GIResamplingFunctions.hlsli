
/**
 * This file is adapted from the original file in the RTXDI SDK.
 * @url: https://github.com/NVIDIAGameWorks/RTXDI/blob/main/shaders/LightingPasses/GITemporalResampling.hlsl
 * The copyright of original file is retained here:
 * /***************************************************************************
 *  # Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *  #
 *  # NVIDIA CORPORATION and its licensors retain all intellectual property
 *  # and proprietary rights in and to this software, related documentation
 *  # and any modifications thereto.  Any use, reproduction, disclosure or
 *  # distribution of this software and related documentation without an express
 *  # license agreement from NVIDIA CORPORATION is strictly prohibited.
 *  **************************************************************************
 */

#ifndef GI_RESAMPLING_FUNCTIONS_HLSLI
#define GI_RESAMPLING_FUNCTIONS_HLSLI

#include "../../../include/common/random.hlsli"
#include "../../../include/common/shading.hlsli"
#include "../../../include/common/microfacet.hlsli"
#include "../../../raytracer/spt_interface.hlsli"
#include "../../gbuffer/gbuffer_common.hlsli"
#include "../../gbuffer/gbuffer_prev_interface.hlsli"
#include "GIReservoir.hlsli"

struct GITemporalResamplingParameters {
    // Screen-space motion vector, computed as (previousPosition - currentPosition).
    // The X and Y components are measured in pixels.
    // The Z component is in linear depth units.
    float3 screenSpaceMotion;
    // The index of the reservoir buffer to pull the temporal samples from.
    uint sourceBufferIndex;
    // Maximum history length for reuse, measured in frames.
    // Higher values result in more stable and high quality sampling, at the cost of slow reaction to changes.
    uint maxHistoryLength;
    // Controls the bias correction math for temporal reuse. Depending on the setting, it can add
    // some shader cost and one approximate shadow ray per pixel (or per two pixels if checkerboard sampling is enabled).
    // Ideally, these rays should be traced through the previous frame's BVH to get fully unbiased results.
    uint biasCorrectionMode;
    // Surface depth similarity threshold for temporal reuse.
    // If the previous frame surface's depth is within this threshold from the current frame surface's depth,
    // the surfaces are considered similar. The threshold is relative, i.e. 0.1 means 10% of the current depth.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float depthThreshold;
    // Surface normal similarity threshold for temporal reuse.
    // If the dot product of two surfaces' normals is higher than this threshold, the surfaces are considered similar.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float normalThreshold;
    // Discard the reservoir if its age exceeds this value.
    uint maxReservoirAge;
    // Enables permuting the pixels sampled from the previous frame in order to add temporal
    // variation to the output signal and make it more denoiser friendly.
    bool enablePermutationSampling;
    // Enables resampling from a location around the current pixel instead of what the motion vector points at,
    // in case no surface near the motion vector matches the current surface (e.g. disocclusion).
    // This behavoir makes disocclusion areas less noisy but locally biased, usually darker.
    bool enableFallbackSampling;
};

// A structure that groups the application-provided settings for spatial resampling.
struct GISpatialResamplingParameters {
    // The index of the reservoir buffer to pull the spatial samples from.
    uint sourceBufferIndex;
    // Surface depth similarity threshold for temporal reuse.
    // If the previous frame surface's depth is within this threshold from the current frame surface's depth,
    // the surfaces are considered similar. The threshold is relative, i.e. 0.1 means 10% of the current depth.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float depthThreshold;
    // Surface normal similarity threshold for temporal reuse.
    // If the dot product of two surfaces' normals is higher than this threshold, the surfaces are considered similar.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float normalThreshold;
    // Number of neighbor pixels considered for resampling (1-32)
    // Some of the may be skipped if they fail the surface similarity test.
    uint numSamples;
    // Screen-space radius for spatial resampling, measured in pixels.
    float samplingRadius;
    // Controls the bias correction math for temporal reuse. Depending on the setting, it can add
    // some shader cost and one approximate shadow ray per pixel (or per two pixels if checkerboard sampling is enabled).
    // Ideally, these rays should be traced through the previous frame's BVH to get fully unbiased results.
    uint biasCorrectionMode;
};

/**
 * Adds `newReservoir` into `reservoir`, returns true if the new reservoir's sample was selected.
 * This function assumes the newReservoir has been normalized, so its weightSum means "1/g * 1/M * \sum{g/p}"
 * and the targetPdf is a conversion factor from the newReservoir's space to the reservoir's space (integrand).
 */
bool CombineGIReservoirs(
    inout_ref(GIReservoir) reservoir,
    in_ref(GIReservoir) newReservoir,
    float random,
    float targetPdf
) {
    // What's the current weight (times any prior-step RIS normalization factor)
    const float risWeight = targetPdf * newReservoir.weightSum * newReservoir.M;
    // Our *effective* candidate pool is the sum of our candidates plus those of our neighbors
    reservoir.M += newReservoir.M;
    // Update the weight sum
    reservoir.weightSum += risWeight;
    // Decide if we will randomly pick this sample
    bool selectSample = (random * reservoir.weightSum <= risWeight);
    if (selectSample) {
        reservoir.position = newReservoir.position;
        reservoir.normal = newReservoir.normal;
        reservoir.radiance = newReservoir.radiance;
        reservoir.age = newReservoir.age;
    }
    return selectSample;
}

/**
 * Performs normalization of the reservoir after streaming.
 * Essentially, after invoking this function, the reservoir's
 * weightSum field will be the 'Unbiased Contribution Weight' Wx.
 * With the formulation of RIS, it is calculated as follows:
 *      {wSum * (1/ M)} * 1/selectedTargetPdf
 * We use this formulation, but notice that in GRIS formulation,
 * The 1/M is moved into wi, so the formula is kind of different.
 * @param normalizationNumerator The numerator of the normalization factor.
 * @param normalizationDenominator The denominator of the normalization factor.
 * generally we could say : normalizationDenominator = M * selectedTargetPdf
 * And if the denominator is 0, the reservoir is invalid and has 0 weightSum.
 */
void FinalizeGIResampling(
    inout_ref(GIReservoir) reservoir,
    float normalizationNumerator,
    float normalizationDenominator
) {
    reservoir.weightSum = (normalizationDenominator == 0.0) ? 0.0 
        : (reservoir.weightSum * normalizationNumerator) / normalizationDenominator;
}

/**
 * Compares two values and returns true if their relative difference is lower than the threshold.
 * Zero or negative threshold makes test always succeed, not fail.
 * @param reference The reference value.
 * @param candidate The candidate value.
 * @param threshold The threshold.
 */
bool CompareRelativeDifference(float reference, float candidate, float threshold) {
    return (threshold <= 0) || abs(reference - candidate) <= threshold * max(reference, candidate);
}

/**
 * See if we will reuse this neighbor or history sample using
 * edge-stopping functions (e.g., per a bilateral filter).
 * @param ourNorm Our surface normal.
 * @param theirNorm The neighbor surface normal.
 * @param ourDepth Our surface depth.
 */
bool IsValidNeighbor(
    in_ref(float3) ourNorm,
    in_ref(float3) theirNorm, 
    float ourDepth, float theirDepth, 
    float normalThreshold, float depthThreshold
) {
    return (dot(theirNorm.xyz, ourNorm.xyz) >= normalThreshold)
        && CompareRelativeDifference(ourDepth, theirDepth, depthThreshold);
}

struct SplitBrdf {
    float demodulatedDiffuse;
    float3 specular;
};

SplitBrdf EvaluateBrdf(in_ref(ShadingSurface) surface, float3 samplePosition) {
    float3 N = surface.geometryNormal;
    float3 V = surface.viewDir;
    float3 L = normalize(samplePosition - surface.worldPos);
    
    SplitBrdf brdf;
    brdf.demodulatedDiffuse = k_inv_pi * saturate(dot(surface.geometryNormal, L));
    brdf.specular = float3(0);
    // if (surface.roughness == 0)
    //     brdf.specular = 0;
    // else
    //     brdf.specular = GGX_times_NdotL(V, L, surface.normal, max(surface.roughness, kMinRoughness), surface.specularF0);
    return brdf;
}

/**
 * Computes the weight of the given GI sample when
 * the given surface is shaded using that GI sample.
 * Here, we omit the visibility and only use the BRDF.
 * @param samplePosition The position of the GI sample.
 * @param sampleRadiance The radiance of the GI sample.
 * @param surface The surface to shade.
 */
float GetGISampleTargetPdfForSurface(
    in_ref(float3) samplePosition,
    in_ref(float3) sampleRadiance, 
    in_ref(ShadingSurface) surface
) {
    SplitBrdf brdf = EvaluateBrdf(surface, samplePosition);
    float3 reflectedRadiance = sampleRadiance * (brdf.demodulatedDiffuse * surface.diffuseAlbedo + brdf.specular);
    return luminance(reflectedRadiance);
}

/**
 * Generates a pattern of offsets for looking closely around a given pixel.
 * The pattern places 'sampleIdx' at the following locations in screen space around pixel (x):
 *   0 4 3
 *   6 x 7
 *   2 5 1
 */
int2 CalculateTemporalResamplingOffset(int sampleIdx, int radius) {
    sampleIdx &= 7;
    int mask2 = sampleIdx >> 1 & 0x01;       // 0, 0, 1, 1, 0, 0, 1, 1
    int mask4 = 1 - (sampleIdx >> 2 & 0x01); // 1, 1, 1, 1, 0, 0, 0, 0
    int tmp0 = -1 + 2 * (sampleIdx & 0x01);  // -1, 1,....
    int tmp1 = 1 - 2 * mask2;                // 1, 1,-1,-1, 1, 1,-1,-1
    int tmp2 = mask4 | mask2;                // 1, 1, 1, 1, 0, 0, 1, 1
    int tmp3 = mask4 | (1 - mask2);          // 1, 1, 1, 1, 1, 1, 0, 0
    return int2(tmp0, tmp0 * tmp1) * int2(tmp2, tmp3) * radius;
}

/** Internal SDK function that permutes the pixels sampled from the previous frame. */
void ApplyPermutationSampling(inout int2 prevPixelPos, uint uniformRandomNumber) {
    const int2 offset = int2(uniformRandomNumber & 3, (uniformRandomNumber >> 2) & 3);
    prevPixelPos += offset;
    prevPixelPos.x ^= 3;
    prevPixelPos.y ^= 3;
    prevPixelPos -= offset;
}

// Compares the materials of two surfaces, returns true if the surfaces
// are similar enough that we can share the light reservoirs between them.
// If unsure, just return true.
bool AreMaterialsSimilar(in_ref(ShadingSurface) a, in_ref(ShadingSurface) b) {
    const float roughnessThreshold = 0.5;
    const float reflectivityThreshold = 0.25;
    const float albedoThreshold = 0.25;
    if (!CompareRelativeDifference(a.roughness, b.roughness, roughnessThreshold))
        return false;
    if (abs(luminance(a.specularF0) - luminance(b.specularF0)) > reflectivityThreshold)
        return false;
    if (abs(luminance(a.diffuseAlbedo) - luminance(b.diffuseAlbedo)) > albedoThreshold)
        return false;
    return true;
}

// Calculate the elements of the Jacobian to transform the sample's solid angle.
void CalculatePartialJacobian(
    in_ref(float3) recieverPos,
    in_ref(float3) samplePos,
    in_ref(float3) sampleNormal,
    out_ref(float) distanceToSurface, 
    out_ref(float) cosineEmissionAngle
) {
    float3 vec = recieverPos - samplePos;
    distanceToSurface = length(vec);
    cosineEmissionAngle = saturate(dot(sampleNormal, vec / distanceToSurface));
}

// Calculates the full Jacobian for resampling neighborReservoir into a new receiver surface
float CalculateJacobian(
    in_ref(float3) recieverPos,
    in_ref(float3) neighborReceiverPos,
    in_ref(GIReservoir) neighborReservoir,
    out_ref(float4) debug
) {
    // Calculate Jacobian determinant to adjust weight.
    // See Equation (11) in the ReSTIR GI paper.
    float originalDistance;
    float originalCosine;
    float newDistance;
    float newCosine;
    CalculatePartialJacobian(recieverPos, neighborReservoir.position, neighborReservoir.normal, newDistance, newCosine);
    CalculatePartialJacobian(neighborReceiverPos, neighborReservoir.position, neighborReservoir.normal, originalDistance, originalCosine);
    float jacobian = (newCosine * originalDistance * originalDistance)
        / (originalCosine * newDistance * newDistance);
    if (isinf(jacobian) || isnan(jacobian))
        jacobian = 1;
    return jacobian;
}

// Check if the sample is fine to be used as a valid spatial sample.
// This function also be able to clamp the value of the Jacobian.
bool ValidateGISampleWithJacobian(inout float jacobian) {
    // Sold angle ratio is too different. Discard the sample.
    if (jacobian > 10.0 || jacobian < 1 / 10.0) {
        return false;
    }
    // clamp Jacobian.
    jacobian = clamp(jacobian, 1 / 3.0, 3.0);
    return true;
}

// Temporal resampling for GI reservoir pass.
GIReservoir GITemporalResampling(
    in_ref(uint2) pixelPosition,
    in_ref(ShadingSurface) surface,
    in_ref(GIReservoir) inputReservoir,
    in_ref(GITemporalResamplingParameters) tparams,
    in_ref(GIResamplingRuntimeParameters) params,
    in_ref(CameraData) prev_camera,
    in_ref(RWStructuredBuffer<PackedGIReservoir>) reservoir_buffer,
    inout_ref(RandomSamplerState) RNG,
    RWTexture2D<float4> u_debug
) {
    // Backproject this pixel to last frame
    int2 prevPos = int2(round(float2(pixelPosition) + tparams.screenSpaceMotion.xy));
    const float expectedPrevLinearDepth = surface.viewDepth + tparams.screenSpaceMotion.z;
    const int radius = 1;

    GIReservoir temporalReservoir;
    bool foundTemporalReservoir = false;

    const int temporalSampleStartIdx = int(GetNextRandom(RNG) * 8);

    ShadingSurface temporalSurface = EmptyShadingSurface();

    // Try to find a matching surface in the neighborhood of the reprojected pixel
    const int temporalSampleCount = 5;
    const int sampleCount = temporalSampleCount + (tparams.enableFallbackSampling ? 1 : 0);
    for (int i = 0; i < sampleCount; i++) {
        const bool isFirstSample = i == 0;
        const bool isFallbackSample = i == temporalSampleCount;

        int2 offset = int2(0, 0);
        if (isFallbackSample) {
            // Last sample is a fallback for disocclusion areas: use zero motion vector.
            prevPos = int2(pixelPosition);
        }
        else if (!isFirstSample) {
            offset = CalculateTemporalResamplingOffset(temporalSampleStartIdx + i, radius);
        }

        int2 idx = prevPos + offset;
        // if ((tparams.enablePermutationSampling && isFirstSample) || isFallbackSample) {
        //     // Apply permutation sampling for the first (non-jittered) sample,
        //     // also for the last (fallback) sample to prevent visible repeating patterns in disocclusions.
        //     ApplyPermutationSampling(idx, params.uniformRandomNumber);
        // }
        
        // Grab shading / g-buffer data from last frame
        temporalSurface = GetPrevGBufferSurface(idx, prev_camera);
        // skip the sample if the surface is invalid
        if (!IsShadingSurfaceValid(temporalSurface)) {
            continue;
        }

        // Test surface similarity, discard the sample if the surface is too different.
        // Skip this test for the last (fallback) sample.
        if (!isFallbackSample && !IsValidNeighbor(
                                     surface.geometryNormal, temporalSurface.geometryNormal,
                                     expectedPrevLinearDepth, temporalSurface.viewDepth,
                                     tparams.normalThreshold, tparams.depthThreshold))
            continue;

        // Test material similarity and perform any other app-specific tests.
        if (!AreMaterialsSimilar(surface, temporalSurface)) {
            continue;
        }

        // Read temporal reservoir.
        temporalReservoir = LoadGIReservoir(params, idx, tparams.sourceBufferIndex, reservoir_buffer);
        
        // Check if the reservoir is a valid one.
        if (!IsValidGIReservoir(temporalReservoir)) {
            continue;
        }

        foundTemporalReservoir = true;
        break;
    }

    GIReservoir curReservoir = EmptyGIReservoir();

    // Combine the input reservoir into the current reservoir.
    float selectedTargetPdf = 0;
    if (IsValidGIReservoir(inputReservoir)) {
        selectedTargetPdf = GetGISampleTargetPdfForSurface(inputReservoir.position, inputReservoir.radiance, surface);
        CombineGIReservoirs(curReservoir, inputReservoir, /* random = */ 0.5, selectedTargetPdf);
    }

    float4 debug = float4(0, 0, 0, 1);
    if (foundTemporalReservoir) {
        // Found a valid temporal surface and its GI reservoir.
        // Calculate Jacobian determinant to adjust weight.
        float jacobian = CalculateJacobian(
            surface.worldPos, temporalSurface.worldPos, temporalReservoir, debug);

        if (!ValidateGISampleWithJacobian(jacobian))
            foundTemporalReservoir = false;

        temporalReservoir.weightSum *= jacobian;
        // Clamp history length
        temporalReservoir.M = min(temporalReservoir.M, 20);
        // Make the sample older
        ++temporalReservoir.age;
        // discard if the reservoir is too old
        // My experience is discarding based on age make bias.
        // Therefore I discard base on M instead.
        if (temporalReservoir.M > tparams.maxReservoirAge) {
            foundTemporalReservoir = false;
        }
    }

    // // Combine the temporal reservoir into the current reservoir.
    // bool selectedPreviousSample = false;
    if (foundTemporalReservoir) {
        // Reweighting and denormalize the temporal sample with the current surface.
        float targetPdf = GetGISampleTargetPdfForSurface(temporalReservoir.position, temporalReservoir.radiance, surface);
        // Combine the temporalReservoir into the curReservoir
        if (CombineGIReservoirs(curReservoir, temporalReservoir, GetNextRandom(RNG), targetPdf)) {
            selectedTargetPdf = targetPdf;
        }
    }

// #if RTXDI_GI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_BASIC
//     if (tparams.biasCorrectionMode >= RTXDI_BIAS_CORRECTION_BASIC)
//     {
//         float pi = selectedTargetPdf;
//         float piSum = selectedTargetPdf * inputReservoir.M;

//         if (RTXDI_IsValidGIReservoir(curReservoir) && foundTemporalReservoir)
//         {
//             float temporalP = RAB_GetGISampleTargetPdfForSurface(curReservoir.position, curReservoir.radiance, temporalSurface);

// #if RTXDI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_RAY_TRACED
//             if (tparams.biasCorrectionMode == RTXDI_BIAS_CORRECTION_RAY_TRACED && temporalP > 0)
//             {
//                 if (!RAB_GetTemporalConservativeVisibility(surface, temporalSurface, curReservoir.position))
//                 {
//                     temporalP = 0;
//                 }
//             }
// #endif

//             pi = selectedPreviousSample ? temporalP : pi;
//             piSum += temporalP * temporalReservoir.M;
//         }

//         // Normalizing
//         float normalizationNumerator = pi;
//         float normalizationDenominator = piSum * selectedTargetPdf;
//         FinalizeGIResampling(curReservoir, normalizationNumerator, normalizationDenominator);
//     }
//     else
// #endif
    // Normalizing
    const float normalizationNumerator = 1.0;
    const float normalizationDenominator = selectedTargetPdf * curReservoir.M;
    FinalizeGIResampling(curReservoir, normalizationNumerator, normalizationDenominator);

    return curReservoir;
}

int2 CalculateSpatialResamplingOffset(
    int sampleIdx, float radius,
    in_ref(GIResamplingRuntimeParameters) params,
    in_ref(StructuredBuffer<uint8_t>) neighbor_offsets_buffer
) {
    sampleIdx &= int(params.neighborOffsetMask);
    // [0, 1]
    float2 offset = float2(neighbor_offsets_buffer[sampleIdx * 2 + 0],
                           neighbor_offsets_buffer[sampleIdx * 2 + 1]) / 256;
    offset = offset * 2 - 1;
    return int2(offset * radius);
}

GIReservoir GISpatialResampling(
    in_ref(uint2) pixelPosition,
    in_ref(ShadingSurface) surface,
    in_ref(GIReservoir) inputReservoir,
    in_ref(GISpatialResamplingParameters) sparams,
    in_ref(GIResamplingRuntimeParameters) params,
    in_ref(CameraData) camera,
    inout_ref(RandomSamplerState) RNG,
    in_ref(RWStructuredBuffer<PackedGIReservoir>) reservoir_buffer,
    in_ref(StructuredBuffer<uint8_t>) neighbor_offsets_buffer,
    RWTexture2D<float4> u_debug
) {
    // number of spatial samples to combine
    const uint numSamples = sparams.numSamples;
    // The current reservoir.
    GIReservoir curReservoir = EmptyGIReservoir();
    // Simply add the input reservoir into the current reservoir.
    float selectedTargetPdf = 0;
    if (IsValidGIReservoir(inputReservoir)) {
        selectedTargetPdf = GetGISampleTargetPdfForSurface(inputReservoir.position, inputReservoir.radiance, surface);
        CombineGIReservoirs(curReservoir, inputReservoir, /* random = */ 0.5, selectedTargetPdf);
    }
    
    const int2 viewportSize = getViewportSize(camera);

    // We loop through neighbors twice if bias correction is enabled.
    // Cache the validity / edge-stopping function results for the 2nd time through.
    uint cachedResult = 0;

    // Since we're using our bias correction scheme, we need to remember which light selection we made
    int selected = -1;
    
    const int neighborSampleStartIdx = int(GetNextRandom(RNG) * params.neighborOffsetMask);

    // Walk the specified number of spatial neighbors, resampling using RIS
    for (int i = 0; i < numSamples; ++i) {
        // Get screen-space location of neighbor
        int2 offset = CalculateSpatialResamplingOffset(
            neighborSampleStartIdx + i, sparams.samplingRadius, params, neighbor_offsets_buffer);
        int2 idx = int2(pixelPosition) + offset;
        
        idx = clamp(idx, int2(0), int2(viewportSize - 1));
        ShadingSurface neighborSurface = GetGBufferSurface(idx, camera);

        // Test surface similarity, discard the sample if the surface is too different.
        if (!IsValidNeighbor(
                surface.geometryNormal, neighborSurface.geometryNormal,
                surface.viewDepth, neighborSurface.viewDepth,
                sparams.normalThreshold, sparams.depthThreshold))
            continue;

        // Test material similarity and perform any other app-specific tests.
        if (!AreMaterialsSimilar(surface, neighborSurface)) {
            continue;
        }

        GIReservoir neighborReservoir = LoadGIReservoir(params, idx, sparams.sourceBufferIndex, reservoir_buffer);

        if (!IsValidGIReservoir(neighborReservoir))
            continue;

        // Calculate Jacobian determinant to adjust weight.
        // float jacobian = CalculateJacobian(surface.worldPos, neighborSurface.worldPos, neighborReservoir);
        float jacobian = 1.f;
        // Compute reuse weight.
        float targetPdf = GetGISampleTargetPdfForSurface(neighborReservoir.position, neighborReservoir.radiance, surface);

        // The Jacobian to transform a GI sample's solid angle holds the lengths and angles to the GI sample from the surfaces,
        // that are valuable information to determine if the GI sample should be combined with the current sample's stream.
        // This function also may clamp the value of the Jacobian.
        if (!ValidateGISampleWithJacobian(jacobian)) {
            continue;
        }
        
        // Valid neighbor surface and its GI reservoir. Combine the reservor.
        cachedResult |= (1u << uint(i));

        // Combine
        bool isUpdated = CombineGIReservoirs(curReservoir, neighborReservoir, GetNextRandom(RNG), targetPdf * jacobian);
        if (isUpdated) {
            selected = i;
            selectedTargetPdf = targetPdf;
        }
    }

    if (sparams.biasCorrectionMode >= 1) {
        // Compute the unbiased normalization factor (instead of using 1/M)
        float pi = selectedTargetPdf;
        float piSum = selectedTargetPdf * inputReservoir.M;

        // If the GI reservoir has selected other than the initial sample, the position should be come from the previous frame.
        // However, there is no idea for the previous position of the initial GI reservoir, so it just uses the current position as its previous one.
        // float3 selectedPositionInPreviousFrame = curReservoir.position;

        // We need to walk our neighbors again
        for (int i = 0; i < numSamples; ++i) {
            // If we skipped this neighbor above, do so again.
            if ((cachedResult & (1u << uint(i))) == 0) continue;

            // Get the screen-space location of our neighbor
            int2 idx = int2(pixelPosition) + CalculateSpatialResamplingOffset(
                neighborSampleStartIdx + i, sparams.samplingRadius, params, neighbor_offsets_buffer);

            idx = clamp(idx, int2(0), int2(viewportSize - 1));
            // Load our neighbor's G-buffer and its GI reservoir again.
            ShadingSurface neighborSurface = GetGBufferSurface(idx, camera);

            GIReservoir neighborReservoir = LoadGIReservoir(params, idx, sparams.sourceBufferIndex, reservoir_buffer);

            // Get the PDF of the sample RIS selected in the first loop, above, *at this neighbor*
            float ps = GetGISampleTargetPdfForSurface(curReservoir.position, curReservoir.radiance, neighborSurface);

            // This should be done to correct bias.
            if (sparams.biasCorrectionMode == 2 && ps > 0) {
                const Ray ray = SetupVisibilityRay(surface, curReservoir.position, 0.01);
                if (TraceOccludeRay(ray, RNG, SceneBVH)) {
                    ps = 0;
                }
            }
            // Select this sample for the (normalization) numerator if this particular neighbor pixel
            // was the one we selected via RIS in the first loop, above.
            pi = selected == i ? ps : pi;

            // Add to the sums of weights for the (normalization) denominator
            piSum += ps * neighborReservoir.M;
        }

        // "MIS-like" normalization
        // {wSum * (pi/piSum)} * 1/selectedTargetPdf
        {
            float normalizationNumerator = pi;
            float normalizationDenominator = selectedTargetPdf * piSum;
            FinalizeGIResampling(curReservoir, normalizationNumerator, normalizationDenominator);
        }
    }
    else {
        // Normalization
        // {wSum * (1/ M)} * 1/selectedTargetPdf
        const float normalizationNumerator = 1.0;
        const float normalizationDenominator = curReservoir.M * selectedTargetPdf;
        FinalizeGIResampling(curReservoir, normalizationNumerator, normalizationDenominator);
    }
    return curReservoir;
}

#endif // GI_RESAMPLING_FUNCTIONS_HLSLI