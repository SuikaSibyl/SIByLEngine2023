#ifndef _RESTIR_DIRECTIONAL_LIGHT_HEADER_
#define _RESTIR_DIRECTIONAL_LIGHT_HEADER_

#include "restir_di_def.h"
#include "gbuffer.h"

// DIReservoir::lightData packing index with valid bit
// These constants helps to extract the light index from the lightData field
const uint k_DIReservoir_LightIndexMask = 0x7FFFFFFF;
const uint k_DIReservoir_LightValidBit  = 0x80000000;

// Encoding helper constants for RTXDI_PackedReservoir.mVisibility
const uint RTXDI_PackedReservoir_VisibilityMask = 0x3ffff;  // get the visibility bits (low 18 bits)
const uint RTXDI_PackedReservoir_VisibilityChannelMax = 0x3f;   // 6 bits for packing visibility, up to 63
const uint RTXDI_PackedReservoir_VisibilityChannelShift = 6;    // 6 bits for packing visibility
const uint RTXDI_PackedReservoir_MShift = 18;
const uint RTXDI_PackedReservoir_MaxM = 0x3fff; // 14 bits for packing M, up to 16383

// Encoding helper constants for RTXDI_PackedReservoir.distanceAge
const uint RTXDI_PackedReservoir_DistanceChannelBits = 8;   // 8 bits for packing distance
const uint RTXDI_PackedReservoir_DistanceXShift = 0;
const uint RTXDI_PackedReservoir_DistanceYShift = 8;
const uint RTXDI_PackedReservoir_AgeShift = 16;
const uint RTXDI_PackedReservoir_MaxAge = 0xff; // 8 bits for packing age, up to 255
const uint RTXDI_PackedReservoir_DistanceMask = (1u << RTXDI_PackedReservoir_DistanceChannelBits) - 1;
// as we use 8 bits for distance (1 bit for sign), the max distance range is [-127, 127].
const  int RTXDI_PackedReservoir_MaxDistance = int((1u << (RTXDI_PackedReservoir_DistanceChannelBits - 1)) - 1);

/****************************************************************************
* Reservoir data structure creation, packing and storage
* ---------------------------------------------------------------------------
* This section includes functions to create, pack and store reservoirs.
/****************************************************************************/
/** check if the reservoir is valid */
bool IsValidReservoir(const DIReservoir reservoir) {
    return reservoir.lightData != 0; // we pack valid bit into lightData
}
/** get the light index from the reservoir */
uint GetReservoirLightIndex(const DIReservoir reservoir) {
    return reservoir.lightData & k_DIReservoir_LightIndexMask; // remove valid bit
}
/** create a new empty reservoir */
DIReservoir EmptyDIReservoir() {
    DIReservoir s; s.age = 0; s.canonicalWeight = 0;
    s.lightData = 0; s.uvData = 0; s.targetPdf = 0;
    s.weightSum = 0; s.M = 0; s.packedVisibility = 0;
    s.spatialDistance = ivec2(0, 0);
    return s;
}
/** Pack the DIReservoir structure into DIReservoirPacked structure */
DIReservoirPacked PackReservoir(in const DIReservoir reservoir) {
    // Clamp the spatial distance and age. Use 16 bits for distance and 8 bits for age.
    // Spatial distance is clamped to [-127, 127], age is clamped to [0, 255]
    const ivec2 clampedSpatialDistance = clamp(reservoir.spatialDistance, 
        -RTXDI_PackedReservoir_MaxDistance, RTXDI_PackedReservoir_MaxDistance);
    const uint clampedAge = clamp(reservoir.age, 0, RTXDI_PackedReservoir_MaxAge);
    // Create the packed data
    DIReservoirPacked data; data.lightData = reservoir.lightData; data.uvData = reservoir.uvData;
    data.targetPdf = reservoir.targetPdf; data.weight = reservoir.weightSum;
    data.mVisibility = reservoir.packedVisibility // low 18 bits for visibility, high 14 bits for M
        | (min(uint(reservoir.M), RTXDI_PackedReservoir_MaxM) << RTXDI_PackedReservoir_MShift);
    data.distanceAge = // low 8 bits for x distance, next 8 bits for y distance, next 8 bits for age
          ((clampedSpatialDistance.x & RTXDI_PackedReservoir_DistanceMask) << RTXDI_PackedReservoir_DistanceXShift) 
        | ((clampedSpatialDistance.y & RTXDI_PackedReservoir_DistanceMask) << RTXDI_PackedReservoir_DistanceYShift) 
        | (clampedAge << RTXDI_PackedReservoir_AgeShift);
    return data;
}
/** Unpack a DIReservoirPacked to DIReservoir */
DIReservoir UnpackReservoir(DIReservoirPacked data) {
    DIReservoir res; res.lightData = data.lightData; res.uvData = data.uvData;
    res.targetPdf = data.targetPdf; res.weightSum = data.weight; res.canonicalWeight = 0.0f;
    res.M = (data.mVisibility >> RTXDI_PackedReservoir_MShift) & RTXDI_PackedReservoir_MaxM;
    res.packedVisibility = data.mVisibility & RTXDI_PackedReservoir_VisibilityMask;
    // Sign extend the shift values
    res.spatialDistance.x = int(data.distanceAge << (32 - RTXDI_PackedReservoir_DistanceXShift
     - RTXDI_PackedReservoir_DistanceChannelBits)) >> (32 - RTXDI_PackedReservoir_DistanceChannelBits);
    res.spatialDistance.y = int(data.distanceAge << (32 - RTXDI_PackedReservoir_DistanceYShift 
     - RTXDI_PackedReservoir_DistanceChannelBits)) >> (32 - RTXDI_PackedReservoir_DistanceChannelBits);
    res.age = (data.distanceAge >> RTXDI_PackedReservoir_AgeShift) & RTXDI_PackedReservoir_MaxAge;
    // Discard reservoirs that have Inf/NaN
    if (isinf(res.weightSum) || isnan(res.weightSum)) res = EmptyDIReservoir();
    return res;
}
/** Get the index in the buffer given a 2d position.
* Where the reservoirs are packed in block-linear layout. */
uint RTXDI_ReservoirPositionToPointer(
    in const ReSTIR_DI_ResamplingRuntimeParameters params,
    in const uvec2 reservoirPosition,
    in const uint reservoirArrayIndex
) {
    const uvec2 blockIdx = reservoirPosition / RTXDI_RESERVOIR_BLOCK_SIZE;
    const uvec2 positionInBlock = reservoirPosition % RTXDI_RESERVOIR_BLOCK_SIZE;
    return reservoirArrayIndex * params.reservoirArrayPitch
        + blockIdx.y * params.reservoirBlockRowPitch
        + blockIdx.x * (RTXDI_RESERVOIR_BLOCK_SIZE * RTXDI_RESERVOIR_BLOCK_SIZE)
        + positionInBlock.y * RTXDI_RESERVOIR_BLOCK_SIZE
        + positionInBlock.x;
}
/** Pack and store a reservoir */
void StoreReservoir(
    in const DIReservoir reservoir,
    in const ReSTIR_DI_ResamplingRuntimeParameters params,
    in const uvec2 reservoirPosition,
    in const uint reservoirArrayIndex
) {
    const uint pointer = RTXDI_ReservoirPositionToPointer(params, reservoirPosition, reservoirArrayIndex);
    RTXDI_LIGHT_RESERVOIR_BUFFER[pointer] = PackReservoir(reservoir);
}
/** Load and unpack a reservoir */
DIReservoir LoadReservoir(
    in const ReSTIR_DI_ResamplingRuntimeParameters params,
    in const uvec2 reservoirPosition,
    in const uint reservoirArrayIndex
) {
    uint pointer = RTXDI_ReservoirPositionToPointer(params, reservoirPosition, reservoirArrayIndex);
    return UnpackReservoir(RTXDI_LIGHT_RESERVOIR_BUFFER[pointer]);
}
/** Find the corresponding pixel given a reservoir position.
* Reservoirs are stored in a 2D texture, with each pixel containing a reservoir.
* But it is different than pixel pos when we are using chekerboard.*/
uvec2 ReservoirPosToPixelPos(
    in const uvec2 reservoirIndex,
    in const ReSTIR_DI_ResamplingRuntimeParameters params
) {
    if (params.activeCheckerboardField == 0) return reservoirIndex;
    uvec2 pixelPosition = uvec2(reservoirIndex.x << 1, reservoirIndex.y);
    pixelPosition.x += ((pixelPosition.y + params.activeCheckerboardField) & 1);
    return pixelPosition;
}
/** Stores the visibility term in a compressed form in the reservoir. 
* This function should be called when a shadow ray is cast between a surface and a light sample
* in the initial or final shading passes.
* The `discardIfInvisible` parameter controls whether the reservoir should be reset to an invalid state
* if the visibility is zero, which reduces noise;
* it's safe to use that for the initial samples, but discarding samples when their final visibility is zero
* may result in darkening bias.
*/
void StoreVisibilityInReservoir(
    inout DIReservoir reservoir,
    in const vec3 visibility,
    in const bool discardIfInvisible
) {
    reservoir.packedVisibility = uint(clamp(visibility.x, 0., 1.) * RTXDI_PackedReservoir_VisibilityChannelMax) 
        | (uint(clamp(visibility.y, 0., 1.) * RTXDI_PackedReservoir_VisibilityChannelMax))
            << RTXDI_PackedReservoir_VisibilityChannelShift
        | (uint(clamp(visibility.z, 0., 1.) * RTXDI_PackedReservoir_VisibilityChannelMax))
            << (RTXDI_PackedReservoir_VisibilityChannelShift * 2);
    reservoir.spatialDistance = ivec2(0, 0);
    reservoir.age = 0;
    if (discardIfInvisible && visibility.x == 0 && visibility.y == 0 && visibility.z == 0) {
        // Keep M for correct resampling, remove the actual sample
        reservoir.lightData = 0;
        reservoir.weightSum = 0;
    }
}

DILightSample EmptyLightSample() {
    DILightSample lightSample;
    lightSample.position = vec3(0);
    lightSample.normal  = vec3(0);
    lightSample.radiance = vec3(0);
    lightSample.solidAnglePdf = 0;
    return lightSample;
}

/****************************************************************************
* Light sampling strategies
* ---------------------------------------------------------------------------
* This section includes functions to sample and resample lights.
/****************************************************************************/

/**
* Performs importance sampling of a set of items with their PDF values stored in a 2D texture mipmap.
* The texture must have power-of-2 dimensions and a mip chain up to 2x2 pixels (or 2x1 or 1x2 if the texture is rectangular).
* The mip chain must be generated using a regular 2x2 box filter, which means any standard way of generating a mipmap should work.
*/
void SamplePdfMipmap(
    inout RandomSamplerState rng, 
    sampler2D pdfTexture,           // full mip chain starting from unnormalized sampling pdf in mip 0
    in const uvec2 pdfTextureSize,  // dimensions of pdfTexture at mip 0; must be 16k or less
    out uvec2 position,
    out float pdf
) {
    const int lastMipLevel = max(0, int(floor(log2(max(pdfTextureSize.x, pdfTextureSize.y)))) - 1);
    position = uvec2(0, 0);
    pdf = 1.0;
    for (int mipLevel = lastMipLevel; mipLevel >= 0; --mipLevel) {
        position *= 2;
        vec4 samples;
        samples.x = max(0, texelFetch(pdfTexture, ivec2(position.x + 0, position.y + 0), mipLevel).x);
        samples.y = max(0, texelFetch(pdfTexture, ivec2(position.x + 0, position.y + 1), mipLevel).x);
        samples.z = max(0, texelFetch(pdfTexture, ivec2(position.x + 1, position.y + 0), mipLevel).x);
        samples.w = max(0, texelFetch(pdfTexture, ivec2(position.x + 1, position.y + 1), mipLevel).x);
        const float weightSum = samples.x + samples.y + samples.z + samples.w;
        if (weightSum <= 0) {
            pdf = 0;
            return;
        }
        samples /= weightSum;
        float rnd = GetNextRandom(rng);
        
        if (rnd < samples.x) { 
            pdf *= samples.x;
        } else {
            rnd -= samples.x;
            if (rnd < samples.y) {
                position += uvec2(0, 1);
                pdf *= samples.y;
            } else {
                rnd -= samples.y;
                if (rnd < samples.z) {
                    position += uvec2(1, 0);
                    pdf *= samples.z;
                } else {
                    position += uvec2(1, 1);
                    pdf *= samples.w;
                }
            }
        }
    }
}

/**
* Selects one local light using the provided PDF texture and stores its information in the RIS buffer
* at the position identified by the tileIndex and sampleInTile parameters.
* Additionally, stores compact light information in the companion buffer that is managed by the application,
* through the RAB_StoreCompactLightInfo function.
*/
void PresampleLocalLights(
    inout RandomSamplerState rng, 
    sampler2D pdfTexture,
    in const uvec2 pdfTextureSize,
    in const uint tileIndex,
    in const uint sampleInTile,
    in const ReSTIR_DI_ResamplingRuntimeParameters params
) {
    // sample a light from the pdf texture
    uvec2 texelPosition;
    float pdf;
    SamplePdfMipmap(rng, pdfTexture, pdfTextureSize, texelPosition, pdf);
    const uint lightIndex = ZCurve2DToMortonCode(texelPosition);
    uint risBufferPtr = sampleInTile + tileIndex * params.risBufferParams.tileSize;
    bool compact = false;
    float invSourcePdf = 0;
    if (pdf > 0) {
        invSourcePdf = 1.0 / pdf;
        const LightInfo lightInfo = LoadLightInfo(lightIndex + params.localLightParams.firstLocalLight, false);
        compact = StoreCompactLightInfo(risBufferPtr, lightInfo);
    }
    lightIndex += params.localLightParams.firstLocalLight;
    if(compact) {
        lightIndex |= RTXDI_LIGHT_COMPACT_BIT;
    }
    // Store the index of the light that we found and its inverse pdf.
    // Or zero and zero if we somehow found nothing.
    RTXDI_RIS_BUFFER[risBufferPtr] = uvec2(lightIndex, asuint(invSourcePdf));
}

// // Adds `newReservoir` into `reservoir`, returns true if the new reservoir's sample was selected.
// // This is a very general form, allowing input parameters to specfiy normalization and targetPdf
// // rather than computing them from `newReservoir`.  Named "internal" since these parameters take
// // different meanings (e.g., in RTXDI_CombineReservoirs() or RTXDI_StreamNeighborWithPairwiseMIS())
// bool DIInternalSimpleResample(
//     inout DIReservoir reservoir,
//     const DIReservoir newReservoir,
//     float random,
//     float targetPdf,            // Usually closely related to the sample normalization, 
//     float sampleNormalization,  //     typically off by some multiplicative factor 
//     float sampleM               // In its most basic form, should be newReservoir.M
// ) {
//     // What's the current weight (times any prior-step RIS normalization factor)
//     float risWeight = targetPdf * sampleNormalization;
//     // Our *effective* candidate pool is the sum of our candidates plus those of our neighbors
//     reservoir.M += sampleM;
//     // Update the weight sum
//     reservoir.weightSum += risWeight;
//     // Decide if we will randomly pick this sample
//     const bool selectSample = (random * reservoir.weightSum < risWeight);
//     // If we did select this sample, update the relevant data
//     if (selectSample) {
//         reservoir.lightData = newReservoir.lightData;
//         reservoir.uvData = newReservoir.uvData;
//         reservoir.targetPdf = targetPdf;
//         reservoir.packedVisibility = newReservoir.packedVisibility;
//         reservoir.spatialDistance = newReservoir.spatialDistance;
//         reservoir.age = newReservoir.age;
//     }
//     return selectSample;
// }

// bool CombineDIReservoirs(
//     inout DIReservoir reservoir,
//     in const DIReservoir newReservoir,
//     float random,
//     float targetPdf
// ) {
//     return DIInternalSimpleResample(
//         reservoir,
//         newReservoir,
//         random,
//         targetPdf,
//         newReservoir.weightSum * newReservoir.M,
//         newReservoir.M
//     );
// }

// // Performs normalization of the reservoir after streaming. Equation (6) from the ReSTIR paper.
// void FinalizeDIResampling(
//     inout DIReservoir reservoir,
//     float normalizationNumerator,
//     float normalizationDenominator
// ) {
//     const float denominator = reservoir.targetPdf * normalizationDenominator;
//     reservoir.weightSum = (denominator == 0.0) ? 0.0 : (reservoir.weightSum * normalizationNumerator) / denominator;
// }

// void RTXDI_PresampleEnvironmentMap(
//     inout RAB_RandomSamplerState rng, 
//     RTXDI_TEX2D pdfTexture,
//     uvec2 pdfTextureSize,
//     uint tileIndex,
//     uint sampleInTile,
//     RTXDI_EnvironmentLightRuntimeParameters params)
// {
//     uvec2 texelPosition;
//     float pdf;
//     RTXDI_SamplePdfMipmap(rng, pdfTexture, pdfTextureSize, texelPosition, pdf);

//     // Uniform sampling inside the pixels
//     float2 fPos = float2(texelPosition);
//     fPos.x += RAB_GetNextRandom(rng);
//     fPos.y += RAB_GetNextRandom(rng);
    
//     // Convert texel position to UV and pack it
//     float2 uv = fPos / float2(pdfTextureSize);
//     uint packedUv = uint(saturate(uv.x) * 0xffff) | (uint(saturate(uv.y) * 0xffff) << 16);

//     // Compute the inverse PDF if we found something
//     float invSourcePdf = (pdf > 0) ? (1.0 / pdf) : 0;

//     // Store the result
//     uint risBufferPtr = params.environmentRisBufferOffset + sampleInTile + tileIndex * params.environmentTileSize;
//     RTXDI_RIS_BUFFER[risBufferPtr] = uvec2(packedUv, asuint(invSourcePdf));
// }


// void RTXDI_RandomlySelectLocalLight(
//     inout RAB_RandomSamplerState rng,
//     uint firstLocalLight,
//     uint numLocalLights,
// #if RTXDI_ENABLE_PRESAMPLING
//     bool useRisBuffer,
//     uint risBufferBase,
//     uint risBufferCount,
// #endif
//     out RAB_LightInfo lightInfo,
//     out uint lightIndex,
//     out float invSourcePdf
// )
// {
//     float rnd = RAB_GetNextRandom(rng);
//     lightInfo = RAB_EmptyLightInfo();
//     bool lightLoaded = false;
// #if RTXDI_ENABLE_PRESAMPLING
//     if (useRisBuffer)
//     {
//         uint risSample = min(uint(floor(rnd * risBufferCount)), risBufferCount - 1);
//         uint risBufferPtr = risSample + risBufferBase;

//         uvec2 tileData = RTXDI_RIS_BUFFER[risBufferPtr];
//         lightIndex = tileData.x & RTXDI_LIGHT_INDEX_MASK;
//         invSourcePdf = asfloat(tileData.y);

//         if ((tileData.x & RTXDI_LIGHT_COMPACT_BIT) != 0)
//         {
//             lightInfo = RAB_LoadCompactLightInfo(risBufferPtr);
//             lightLoaded = true;
//         }
//     }
//     else
// #endif
//     {
//         lightIndex = min(uint(floor(rnd * numLocalLights)), numLocalLights - 1) + firstLocalLight;
//         invSourcePdf = float(numLocalLights);
//     }

//     if (!lightLoaded) {
//         lightInfo = RAB_LoadLightInfo(lightIndex, false);
//     }
// }

// float2 RTXDI_RandomlySelectLocalLightUV(RAB_RandomSamplerState rng)
// {
//     float2 uv;
//     uv.x = RAB_GetNextRandom(rng);
//     uv.y = RAB_GetNextRandom(rng);
//     return uv;
// }

// // Returns false if the blended source PDF == 0, true otherwise
// bool RTXDI_StreamLocalLightAtUVIntoReservoir(
//     inout RAB_RandomSamplerState rng,
//     RTXDI_SampleParameters sampleParams,
//     RAB_Surface surface,
//     uint lightIndex,
//     float2 uv,
//     float invSourcePdf,
//     RAB_LightInfo lightInfo,
//     inout RTXDI_Reservoir state,
//     inout RAB_LightSample o_selectedSample)
// {
//     RAB_LightSample candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, uv);
//     float blendedSourcePdf = RTXDI_LightBrdfMisWeight(surface, candidateSample, 1.0 / invSourcePdf,
//         sampleParams.localLightMisWeight, false, sampleParams);
//     float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
//     float risRnd = RAB_GetNextRandom(rng);

//     if (blendedSourcePdf == 0)
//     {
//         return false;
//     }
//     bool selected = RTXDI_StreamSample(state, lightIndex, uv, risRnd, targetPdf, 1.0 / blendedSourcePdf);

//     if (selected) {
//         o_selectedSample = candidateSample;
//     }
//     return true;
// }

// // A helper used for pairwise MIS computations.  This might be able to simplify code elsewhere, too.
// float TargetPdfHelper(
//     const RTXDI_Reservoir lightReservoir,
//     const RAB_Surface surface,
//     bool priorFrame)
// {
//     RAB_LightSample lightSample = RAB_SamplePolymorphicLight(
//         RAB_LoadLightInfo(RTXDI_GetReservoirLightIndex(lightReservoir), priorFrame),
//         surface, RTXDI_GetReservoirSampleUV(lightReservoir));

//     return RAB_GetLightSampleTargetPdfForSurface(lightSample, surface);
// }

// DIReservoir RTXDI_TemporalResampling(
//     uvec2 pixelPosition,
//     RAB_Surface surface,
//     in const DIReservoir curSample,
//     inout RAB_RandomSamplerState rng,
//     RTXDI_TemporalResamplingParameters tparams,
//     ReSTIR_DI_ResamplingRuntimeParameters params,
//     out int2 temporalSamplePixelPos,
//     inout RAB_LightSample selectedLightSample)
// {
//     // For temporal reuse, there's only a pair of samples; pairwise and basic MIS are essentially identical
//     if (tparams.biasCorrectionMode == RTXDI_BIAS_CORRECTION_PAIRWISE)
//     {
//         tparams.biasCorrectionMode = RTXDI_BIAS_CORRECTION_BASIC;
//     }

//     uint historyLimit = min(RTXDI_PackedReservoir_MaxM, uint(tparams.maxHistoryLength * curSample.M));

//     int selectedLightPrevID = -1;
//     if (IsValidReservoir(curSample)) {
//         selectedLightPrevID = RAB_TranslateLightIndex(GetReservoirLightIndex(curSample), true);
//     }

//     temporalSamplePixelPos = ivec2(-1, -1);

//     DIReservoir state = EmptyDIReservoir();
//     CombineDIReservoirs(state, curSample, /* random = */ 0.5, curSample.targetPdf);

//     // Backproject this pixel to last frame
//     float3 motion = tparams.screenSpaceMotion;
    
//     if (!tparams.enablePermutationSampling)
//     {
//         motion.xy += float2(RAB_GetNextRandom(rng), RAB_GetNextRandom(rng)) - 0.5;
//     }

//     vec2 reprojectedSamplePosition = vec2(pixelPosition) + motion.xy;
//     ivec2 prevPos = ivec2(round(reprojectedSamplePosition));

//     float expectedPrevLinearDepth = RAB_GetSurfaceLinearDepth(surface) + motion.z;

//     RAB_Surface temporalSurface = RAB_EmptySurface();
//     bool foundNeighbor = false;
//     const float radius = (params.activeCheckerboardField == 0) ? 4 : 8;
//     ivec2 spatialOffset = ivec2(0, 0);

//     // Try to find a matching surface in the neighborhood of the reprojected pixel
//     for(int i = 0; i < 9; i++) {
//         ivec2 offset = ivec2(0, 0);
//         if(i > 0)
//         {
//             offset.x = int((RAB_GetNextRandom(rng) - 0.5) * radius);
//             offset.y = int((RAB_GetNextRandom(rng) - 0.5) * radius);
//         }

//         ivec2 idx = prevPos + offset;
//         if (tparams.enablePermutationSampling && i == 0)
//         {
//             RTXDI_ApplyPermutationSampling(idx, params.uniformRandomNumber);
//         }

//         RTXDI_ActivateCheckerboardPixel(idx, true, params);

//         // Grab shading / g-buffer data from last frame
//         temporalSurface = RAB_GetGBufferSurface(idx, true);
//         if (!RAB_IsSurfaceValid(temporalSurface))
//             continue;
        
//         // Test surface similarity, discard the sample if the surface is too different.
//         if (!RTXDI_IsValidNeighbor(
//             RAB_GetSurfaceNormal(surface), RAB_GetSurfaceNormal(temporalSurface), 
//             expectedPrevLinearDepth, RAB_GetSurfaceLinearDepth(temporalSurface), 
//             tparams.normalThreshold, tparams.depthThreshold))
//             continue;

//         spatialOffset = idx - prevPos;
//         prevPos = idx;
//         foundNeighbor = true;

//         break;
//     }

//     bool selectedPreviousSample = false;
//     float previousM = 0;

//     if (foundNeighbor) {
//         // Resample the previous frame sample into the current reservoir, but reduce the light's weight
//         // according to the bilinear weight of the current pixel
//         uvec2 prevReservoirPos = RTXDI_PixelPosToReservoirPos(prevPos, params);
//         RTXDI_Reservoir prevSample = RTXDI_LoadReservoir(params,
//             prevReservoirPos, tparams.sourceBufferIndex);
//         prevSample.M = min(prevSample.M, historyLimit);
//         prevSample.spatialDistance += spatialOffset;
//         prevSample.age += 1;

//         uint originalPrevLightID = RTXDI_GetReservoirLightIndex(prevSample);

//         // Map the light ID from the previous frame into the current frame, if it still exists
//         if (IsValidReservoir(prevSample))
//         {
//             if (prevSample.age <= 1)
//             {
//                 temporalSamplePixelPos = prevPos;
//             }

//             int mappedLightID = RAB_TranslateLightIndex(RTXDI_GetReservoirLightIndex(prevSample), false);

//             if (mappedLightID < 0)
//             {
//                 // Kill the reservoir
//                 prevSample.weightSum = 0;
//                 prevSample.lightData = 0;
//             }
//             else
//             {
//                 // Sample is valid - modify the light ID stored
//                 prevSample.lightData = mappedLightID | RTXDI_Reservoir_LightValidBit;
//             }
//         }

//         previousM = prevSample.M;

//         float weightAtCurrent = 0;
//         RAB_LightSample candidateLightSample = RAB_EmptyLightSample();
//         if (RTXDI_IsValidReservoir(prevSample))
//         {
//             const RAB_LightInfo candidateLight = RAB_LoadLightInfo(RTXDI_GetReservoirLightIndex(prevSample), false);

//             candidateLightSample = RAB_SamplePolymorphicLight(
//                 candidateLight, surface, RTXDI_GetReservoirSampleUV(prevSample));

//             weightAtCurrent = RAB_GetLightSampleTargetPdfForSurface(candidateLightSample, surface);
//         }

//         bool sampleSelected = RTXDI_CombineReservoirs(state, prevSample, RAB_GetNextRandom(rng), weightAtCurrent);
//         if(sampleSelected)
//         {
//             selectedPreviousSample = true;
//             selectedLightPrevID = int(originalPrevLightID);
//             selectedLightSample = candidateLightSample;
//         }
//     }

// #if RTXDI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_BASIC
//     if (tparams.biasCorrectionMode >= RTXDI_BIAS_CORRECTION_BASIC)
//     {
//         // Compute the unbiased normalization term (instead of using 1/M)
//         float pi = state.targetPdf;
//         float piSum = state.targetPdf * curSample.M;
        
//         if (RTXDI_IsValidReservoir(state) && selectedLightPrevID >= 0 && previousM > 0)
//         {
//             float temporalP = 0;

//             const RAB_LightInfo selectedLightPrev = RAB_LoadLightInfo(selectedLightPrevID, true);

//             // Get the PDF of the sample RIS selected in the first loop, above, *at this neighbor* 
//             const RAB_LightSample selectedSampleAtTemporal = RAB_SamplePolymorphicLight(
//                 selectedLightPrev, temporalSurface, RTXDI_GetReservoirSampleUV(state));
        
//             temporalP = RAB_GetLightSampleTargetPdfForSurface(selectedSampleAtTemporal, temporalSurface);

// #if RTXDI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_RAY_TRACED
//             if (tparams.biasCorrectionMode == RTXDI_BIAS_CORRECTION_RAY_TRACED && temporalP > 0 && (!selectedPreviousSample || !tparams.enableVisibilityShortcut))
//             {
//                 if (!RAB_GetTemporalConservativeVisibility(surface, temporalSurface, selectedSampleAtTemporal))
//                 {
//                     temporalP = 0;
//                 }
//             }
// #endif

//             pi = selectedPreviousSample ? temporalP : pi;
//             piSum += temporalP * previousM;
//         }

//         RTXDI_FinalizeResampling(state, pi, piSum);
//     }
//     else
// #endif
//     {
//         RTXDI_FinalizeResampling(state, 1.0, state.M);
//     }

//     return state;
// }


// // A structure that groups the application-provided settings for spatial resampling.
// struct RTXDI_SpatialResamplingParameters
// {
//     // The index of the reservoir buffer to pull the spatial samples from.
//     uint sourceBufferIndex;
    
//     // Number of neighbor pixels considered for resampling (1-32)
//     // Some of the may be skipped if they fail the surface similarity test.
//     uint numSamples;

//     // Number of neighbor pixels considered when there is not enough history data (1-32)
//     // Setting this parameter equal or lower than `numSpatialSamples` effectively
//     // disables the disocclusion boost.
//     uint numDisocclusionBoostSamples;

//     // Disocclusion boost is activated when the current reservoir's M value
//     // is less than targetHistoryLength.
//     uint targetHistoryLength;


//     // Controls the bias correction math for spatial reuse. Depending on the setting, it can add
//     // some shader cost and one approximate shadow ray *per every spatial sample* per pixel 
//     // (or per two pixels if checkerboard sampling is enabled).
//     uint biasCorrectionMode;

//     // Screen-space radius for spatial resampling, measured in pixels.
//     float samplingRadius;

//     // Surface depth similarity threshold for spatial reuse.
//     // See 'RTXDI_TemporalResamplingParameters::depthThreshold' for more information.
//     float depthThreshold;

//     // Surface normal similarity threshold for spatial reuse.
//     // See 'RTXDI_TemporalResamplingParameters::normalThreshold' for more information.
//     float normalThreshold;

//     // Enables the comparison of surface materials before taking a surface into resampling.
//     bool enableMaterialSimilarityTest;
// };

// // Spatial resampling pass, using pairwise MIS.  
// // Inputs and outputs equivalent to RTXDI_SpatialResampling(), but only uses pairwise MIS.
// // Can call this directly, or call RTXDI_SpatialResampling() with sparams.biasCorrectionMode 
// // set to RTXDI_BIAS_CORRECTION_PAIRWISE, which simply calls this function.
// RTXDI_Reservoir RTXDI_SpatialResamplingWithPairwiseMIS(
//     uvec2 pixelPosition,
//     RAB_Surface centerSurface,
//     RTXDI_Reservoir centerSample,
//     inout RAB_RandomSamplerState rng,
//     RTXDI_SpatialResamplingParameters sparams,
//     ReSTIR_DI_ResamplingRuntimeParameters params,
//     inout RAB_LightSample selectedLightSample)
// {
//     // Initialize the output reservoir
//     RTXDI_Reservoir state = RTXDI_EmptyReservoir();
//     state.canonicalWeight = 0.0f;

//     // How many spatial samples to use?  
//     uint numSpatialSamples = (centerSample.M < sparams.targetHistoryLength)
//         ? max(sparams.numDisocclusionBoostSamples, sparams.numSamples)
//         : sparams.numSamples;

//     // Walk the specified number of neighbors, resampling using RIS
//     uint startIdx = uint(RAB_GetNextRandom(rng) * params.neighborOffsetMask);
//     uint validSpatialSamples = 0;
//     uint i;
//     for (i = 0; i < numSpatialSamples; ++i)
//     {
//         // Get screen-space location of neighbor
//         uint sampleIdx = (startIdx + i) & params.neighborOffsetMask;
//         ivec2 spatialOffset = ivec2(vec2(RTXDI_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * sparams.samplingRadius);
//         ivec2 idx = ivec2(pixelPosition) + spatialOffset;
//         idx = RAB_ClampSamplePositionIntoView(idx, false);

//         RTXDI_ActivateCheckerboardPixel(idx, false, params);

//         RAB_Surface neighborSurface = RAB_GetGBufferSurface(idx, false);

//         // Check for surface / G-buffer matches between the canonical sample and this neighbor
//         if (!RAB_IsSurfaceValid(neighborSurface))
//             continue;

//         if (!RTXDI_IsValidNeighbor(RAB_GetSurfaceNormal(centerSurface), RAB_GetSurfaceNormal(neighborSurface),
//             RAB_GetSurfaceLinearDepth(centerSurface), RAB_GetSurfaceLinearDepth(neighborSurface),
//             sparams.normalThreshold, sparams.depthThreshold))
//             continue;

//         if (sparams.enableMaterialSimilarityTest && !RAB_AreMaterialsSimilar(centerSurface, neighborSurface))
//             continue;

//         // The surfaces are similar enough so we *can* reuse a neighbor from this pixel, so load it.
//         RTXDI_Reservoir neighborSample = RTXDI_LoadReservoir(params,
//             RTXDI_PixelPosToReservoirPos(idx, params), sparams.sourceBufferIndex);
//         neighborSample.spatialDistance += spatialOffset;

//         validSpatialSamples++;

//         // If sample has weight 0 due to visibility (or etc), skip the expensive-ish MIS computations
//         if (neighborSample.M <= 0) continue;

//         // Stream this light through the reservoir using pairwise MIS
//         RTXDI_StreamNeighborWithPairwiseMIS(state, RAB_GetNextRandom(rng),
//             neighborSample, neighborSurface,   // The spatial neighbor
//             centerSample, centerSurface,       // The canonical (center) sample
//             numSpatialSamples);
//     }

//     // If we've seen no usable neighbor samples, set the weight of the central one to 1
//     state.canonicalWeight = (validSpatialSamples <= 0) ? 1.0f : state.canonicalWeight;

//     // Stream the canonical sample (i.e., from prior computations at this pixel in this frame) using pairwise MIS.
//     RTXDI_StreamCanonicalWithPairwiseStep(state, RAB_GetNextRandom(rng), centerSample, centerSurface);

//     RTXDI_FinalizeResampling(state, 1.0, float(max(1, validSpatialSamples)));

//     // Return the selected light sample.  This is a redundant lookup and could be optimized away by storing
//         // the selected sample from the stream steps above.
//     selectedLightSample = RAB_SamplePolymorphicLight(
//         RAB_LoadLightInfo(RTXDI_GetReservoirLightIndex(state), false),
//         centerSurface, RTXDI_GetReservoirSampleUV(state));

//     return state;
// }


// // Spatial resampling pass.
// // Operates on the current frame G-buffer and its reservoirs.
// // For each pixel, considers a number of its neighbors and, if their surfaces are 
// // similar enough to the current pixel, combines their light reservoirs.
// // Optionally, one visibility ray is traced for each neighbor being considered, to reduce bias.
// // The selectedLightSample parameter is used to update and return the selected sample; it's optional,
// // and it's safe to pass a null structure there and ignore the result.
// RTXDI_Reservoir RTXDI_SpatialResampling(
//     uvec2 pixelPosition,
//     RAB_Surface centerSurface,
//     RTXDI_Reservoir centerSample,
//     inout RAB_RandomSamplerState rng,
//     RTXDI_SpatialResamplingParameters sparams,
//     ReSTIR_DI_ResamplingRuntimeParameters params,
//     inout RAB_LightSample selectedLightSample)
// {
//     if (sparams.biasCorrectionMode == RTXDI_BIAS_CORRECTION_PAIRWISE)
//     {
//         return RTXDI_SpatialResamplingWithPairwiseMIS(pixelPosition, centerSurface, 
//             centerSample, rng, sparams, params, selectedLightSample);
//     }

//     RTXDI_Reservoir state = RTXDI_EmptyReservoir();

//     // This is the weight we'll use (instead of 1/M) to make our estimate unbaised (see paper).
//     float normalizationWeight = 1.0f;

//     // Since we're using our bias correction scheme, we need to remember which light selection we made
//     int selected = -1;

//     RAB_LightInfo selectedLight = RAB_EmptyLightInfo();

//     if (RTXDI_IsValidReservoir(centerSample))
//     {
//         selectedLight = RAB_LoadLightInfo(RTXDI_GetReservoirLightIndex(centerSample), false);
//     }

//     RTXDI_CombineReservoirs(state, centerSample, /* random = */ 0.5f, centerSample.targetPdf);

//     uint startIdx = uint(RAB_GetNextRandom(rng) * params.neighborOffsetMask);
    
//     uint i;
//     uint numSpatialSamples = sparams.numSamples;
//     if(centerSample.M < sparams.targetHistoryLength)
//         numSpatialSamples = max(sparams.numDisocclusionBoostSamples, numSpatialSamples);

//     // Clamp the sample count at 32 to make sure we can keep the neighbor mask in an uint (cachedResult)
//     numSpatialSamples = min(numSpatialSamples, 32);

//     // We loop through neighbors twice.  Cache the validity / edge-stopping function
//     //   results for the 2nd time through.
//     uint cachedResult = 0;

//     // Walk the specified number of neighbors, resampling using RIS
//     for (i = 0; i < numSpatialSamples; ++i)
//     {
//         // Get screen-space location of neighbor
//         uint sampleIdx = (startIdx + i) & params.neighborOffsetMask;
//         int2 spatialOffset = int2(float2(RTXDI_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * sparams.samplingRadius);
//         int2 idx = int2(pixelPosition) + spatialOffset;

//         idx = RAB_ClampSamplePositionIntoView(idx, false);

//         RTXDI_ActivateCheckerboardPixel(idx, false, params);

//         RAB_Surface neighborSurface = RAB_GetGBufferSurface(idx, false);

//         if (!RAB_IsSurfaceValid(neighborSurface))
//             continue;

//         if (!RTXDI_IsValidNeighbor(RAB_GetSurfaceNormal(centerSurface), RAB_GetSurfaceNormal(neighborSurface), 
//             RAB_GetSurfaceLinearDepth(centerSurface), RAB_GetSurfaceLinearDepth(neighborSurface), 
//             sparams.normalThreshold, sparams.depthThreshold))
//             continue;

//         if (sparams.enableMaterialSimilarityTest && !RAB_AreMaterialsSimilar(centerSurface, neighborSurface))
//             continue;

//         uvec2 neighborReservoirPos = RTXDI_PixelPosToReservoirPos(idx, params);

//         RTXDI_Reservoir neighborSample = RTXDI_LoadReservoir(params,
//             neighborReservoirPos, sparams.sourceBufferIndex);
//         neighborSample.spatialDistance += spatialOffset;

//         cachedResult |= (1u << uint(i));

//         RAB_LightInfo candidateLight = RAB_EmptyLightInfo();

//         // Load that neighbor's RIS state, do resampling
//         float neighborWeight = 0;
//         RAB_LightSample candidateLightSample = RAB_EmptyLightSample();
//         if (RTXDI_IsValidReservoir(neighborSample))
//         {   
//             candidateLight = RAB_LoadLightInfo(RTXDI_GetReservoirLightIndex(neighborSample), false);
            
//             candidateLightSample = RAB_SamplePolymorphicLight(
//                 candidateLight, centerSurface, RTXDI_GetReservoirSampleUV(neighborSample));
            
//             neighborWeight = RAB_GetLightSampleTargetPdfForSurface(candidateLightSample, centerSurface);
//         }
        
//         if (RTXDI_CombineReservoirs(state, neighborSample, RAB_GetNextRandom(rng), neighborWeight))
//         {
//             selected = int(i);
//             selectedLight = candidateLight;
//             selectedLightSample = candidateLightSample;
//         }
//     }

//     if (RTXDI_IsValidReservoir(state))
//     {
// #if RTXDI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_BASIC
//         if (sparams.biasCorrectionMode >= RTXDI_BIAS_CORRECTION_BASIC)
//         {
//             // Compute the unbiased normalization term (instead of using 1/M)
//             float pi = state.targetPdf;
//             float piSum = state.targetPdf * centerSample.M;

//             // To do this, we need to walk our neighbors again
//             for (i = 0; i < numSpatialSamples; ++i)
//             {
//                 // If we skipped this neighbor above, do so again.
//                 if ((cachedResult & (1u << uint(i))) == 0) continue;

//                 uint sampleIdx = (startIdx + i) & params.neighborOffsetMask;

//                 // Get the screen-space location of our neighbor
//                 int2 idx = int2(pixelPosition) + int2(float2(RTXDI_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * sparams.samplingRadius);

//                 idx = RAB_ClampSamplePositionIntoView(idx, false);

//                 RTXDI_ActivateCheckerboardPixel(idx, false, params);

//                 // Load our neighbor's G-buffer
//                 RAB_Surface neighborSurface = RAB_GetGBufferSurface(idx, false);
                
//                 // Get the PDF of the sample RIS selected in the first loop, above, *at this neighbor* 
//                 const RAB_LightSample selectedSampleAtNeighbor = RAB_SamplePolymorphicLight(
//                     selectedLight, neighborSurface, RTXDI_GetReservoirSampleUV(state));

//                 float ps = RAB_GetLightSampleTargetPdfForSurface(selectedSampleAtNeighbor, neighborSurface);

// #if RTXDI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_RAY_TRACED
//                 if (sparams.biasCorrectionMode == RTXDI_BIAS_CORRECTION_RAY_TRACED && ps > 0)
//                 {
//                     if (!RAB_GetConservativeVisibility(neighborSurface, selectedSampleAtNeighbor))
//                     {
//                         ps = 0;
//                     }
//                 }
// #endif

//                 uvec2 neighborReservoirPos = RTXDI_PixelPosToReservoirPos(idx, params);

//                 RTXDI_Reservoir neighborSample = RTXDI_LoadReservoir(params,
//                     neighborReservoirPos, sparams.sourceBufferIndex);

//                 // Select this sample for the (normalization) numerator if this particular neighbor pixel
//                 //     was the one we selected via RIS in the first loop, above.
//                 pi = selected == i ? ps : pi;

//                 // Add to the sums of weights for the (normalization) denominator
//                 piSum += ps * neighborSample.M;
//             }

//             // Use "MIS-like" normalization
//             RTXDI_FinalizeResampling(state, pi, piSum);
//         }
//         else
// #endif
//         {
//             RTXDI_FinalizeResampling(state, 1.0, state.M);
//         }
//     }

//     return state;
// }

// struct Reservoir {
// 	float y;    // the output sample
// 	float wsum; // the sum of weights
// 	float M;    // the number of samples seen so far
// 	float W;    // probablistic weight
// };

// /**
// * Update the reservoir with a new sample.
// * @param r: The reservoir to update.
// * @param x: The current sample to update.
// * @param w: The weight of the current sample.
// * @param rand: The random number to control the stochastic process.
// */
// void UpdateResrvoir(
//     inout Reservoir r,
//     in const float x,
//     in const float w,
//     in const float rand
// ) {
//     r.wsum += w;
//     r.M += 1;
//     if (rand < (w / r.wsum))
//         r.y = x;
// }

// /**
// */
// float InvalidResrvoirWeight(
//     inout Reservoir r,
//     in const float currentWeight
// ) {
//     r.W = r.wsum / (r.M * currentWeight);
//     return r.W;
// }

// /**
// */
// Reservoir CombineResrvoirs(
//     in const Reservoir r1,
//     in const Reservoir r2,
//     in const float currentWeight
// ) {
//     // r.W = r.wsum / (r.M * currentWeight);
//     // return r.W;
// }


// #ifndef ReSTIRDI_TILE_SIZE_IN_PIXELS
// #define ReSTIRDI_TILE_SIZE_IN_PIXELS 16
// #endif

void RandomlySelectLocalLight(
    inout RAB_RandomSamplerState rng,
    uint firstLocalLight,
    uint numLocalLights,
    bool useRisBuffer,
    uint risBufferBase,
    uint risBufferCount,
    out RAB_LightInfo lightInfo,
    out uint lightIndex,
    out float invSourcePdf
) {
    const float rnd = GetNextRandom(rng);
    lightInfo = EmptyLightInfo();
    bool lightLoaded = false;
    if (useRisBuffer) {
        uint risSample = min(uint(floor(rnd * risBufferCount)), risBufferCount - 1);
        uint risBufferPtr = risSample + risBufferBase;

        uvec2 tileData = RTXDI_RIS_BUFFER[risBufferPtr];
        lightIndex = tileData.x & RTXDI_LIGHT_INDEX_MASK;
        invSourcePdf = asfloat(tileData.y);

        if ((tileData.x & RTXDI_LIGHT_COMPACT_BIT) != 0) {
            lightInfo = RAB_LoadCompactLightInfo(risBufferPtr);
            lightLoaded = true;
        }
    }
    else {
        lightIndex = min(uint(floor(rnd * numLocalLights)), numLocalLights - 1) + firstLocalLight;
        invSourcePdf = float(numLocalLights);
    }

    if (!lightLoaded) {
        lightInfo = LoadLightInfo(lightIndex, false);
    }
}

vec2 RandomlySelectLocalLightUV(inout RandomSamplerState rng) {
    return vec2(GetNextRandom(rng), GetNextRandom(rng));
}

// // Returns false if the blended source PDF == 0, true otherwise
// bool RTXDI_StreamLocalLightAtUVIntoReservoir(
//     inout RAB_RandomSamplerState rng,
//     RTXDI_SampleParameters sampleParams,
//     RAB_Surface surface,
//     uint lightIndex,
//     float2 uv,
//     float invSourcePdf,
//     RAB_LightInfo lightInfo,
//     inout RTXDI_Reservoir state,
//     inout RAB_LightSample o_selectedSample)
// {
//     RAB_LightSample candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, uv);
//     float blendedSourcePdf = RTXDI_LightBrdfMisWeight(surface, candidateSample, 1.0 / invSourcePdf,
//         sampleParams.localLightMisWeight, false, sampleParams);
//     float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
//     float risRnd = RAB_GetNextRandom(rng);

//     if (blendedSourcePdf == 0) {
//         return false;
//     }
//     bool selected = RTXDI_StreamSample(state, lightIndex, uv, risRnd, targetPdf, 1.0 / blendedSourcePdf);

//     if (selected) {
//         o_selectedSample = candidateSample;
//     }
//     return true;
// }

/** SDK internal function that samples the given set of lights generated by RIS
* or the local light pool. The RIS set can come from local light importance presampling or from ReGIR. */
DIReservoir RTXDI_SampleLocalLightsInternal(
    inout RandomSamplerState rng, 
    in const GBufferSurface surface,
    in const ReSTIR_DI_SampleParameters sampleParams,
    in const ReSTIR_DI_ResamplingRuntimeParameters params, 
    bool useRisBuffer,
    uint risBufferBase,
    uint risBufferCount,
    out DILightSample o_selectedSample
) {
    DIReservoir state = EmptyDIReservoir();
    o_selectedSample = EmptyLightSample();

    if (params.numLocalLights == 0)
        return state;

    if (sampleParams.numLocalLightSamples == 0)
        return state;

    for (uint i = 0; i < sampleParams.numLocalLightSamples; ++i) {
        uint lightIndex;
        LightInfo lightInfo;
        float invSourcePdf;

        RandomlySelectLocalLight(rng, params.firstLocalLight, params.numLocalLights,
            useRisBuffer, risBufferBase, risBufferCount,
            lightInfo, lightIndex, invSourcePdf);

        vec2 uv = RandomlySelectLocalLightUV(rng);
        bool zeroPdf = RTXDI_StreamLocalLightAtUVIntoReservoir(rng, sampleParams, surface, lightIndex, uv, invSourcePdf, lightInfo, state, o_selectedSample);
        if (zeroPdf)
            continue;
    }
    
    RTXDI_FinalizeResampling(state, 1.0, sampleParams.numMisSamples);
    state.M = 1;

    return state;
}

/** 
* Samples the local light pool for the given surface.
* 
* Selects one local light sample using RIS with `sampleParams.numLocalLightSamples` 
* proposals weighted relative to the provided `surface`,
* and returns a reservoir with the selected light sample.
* The sample itself is returned in the `o_selectedSample` parameter.
*
* The proposals are picked from a RIS buffer tile that's picked using `coherentRng`, which should generate 
* the same random numbers for a group of adjacent shader threads for performance.
* If the RIS buffer is not available, this function will fall back to uniform sampling from the local light pool,
* which is typically much more noisy. The RIS buffer must be pre-filled with samples using the
* [`RTXDI_PresampleLocalLights`](#rtxdi_presamplelocallights) function in a preceding pass.
*/
DIReservoir SampleLocalLights(
    inout RAB_RandomSamplerState rng, 
    inout RAB_RandomSamplerState coherentRng,
    in const GBufferSurface surface,
    in const ReSTIR_DI_SampleParameters sampleParams,
    in const ReSTIR_DI_ResamplingRuntimeParameters params, 
    out DILightSample o_selectedSample
) {
    const float tileRnd = GetNextRandom(coherentRng);
    const uint tileIndex = uint(tileRnd * params.risBufferParams.tileCount);
    const uint risBufferBase = tileIndex * params.risBufferParams.tileSize;

    return RTXDI_SampleLocalLightsInternal(rng, surface, sampleParams, params.localLightParams,
#if RTXDI_ENABLE_PRESAMPLING
        params.localLightParams.enableLocalLightImportanceSampling != 0, risBufferBase, params.risBufferParams.tileSize,
#endif
        o_selectedSample);
}

/**
* This function is a combination of `RTXDI_SampleInfiniteLights`, `RTXDI_SampleEnvironmentMap`, and `RTXDI_SampleBrdf`
* Reservoirs returned from each function are combined into one final reservoir, which is returned.
* Samples ReGIR and the local and infinite light pools for a given surface. */
DIReservoir RTXDI_SampleLightsForSurface(
    inout RandomSamplerState rng,
    inout RandomSamplerState coherentRng,
    in const GBufferSurface surface,
    in const ReSTIR_DI_SampleParameters sampleParams,
    in const ReSTIR_DI_ResamplingRuntimeParameters params, 
    out DILightSample o_lightSample
) {
    o_lightSample = EmptyLightSample();

    DILightSample localSample = EmptyLightSample();
    const DIReservoir localReservoir = SampleLocalLights(rng, coherentRng, surface, 
        sampleParams, params, localSample);

//     DILightSample infiniteSample = EmptyLightSample();  
//     DIReservoir infiniteReservoir = RTXDI_SampleInfiniteLights(rng, surface,
//         sampleParams.numInfiniteLightSamples, params.infiniteLightParams, infiniteSample);

// #if RTXDI_ENABLE_PRESAMPLING
//     DILightSample environmentSample = EmptyLightSample();
//     DIReservoir environmentReservoir = RTXDI_SampleEnvironmentMap(rng, coherentRng, surface,
//         sampleParams, params.environmentLightParams, environmentSample);
// #endif

//     DILightSample brdfSample = EmptyLightSample();
//     DIReservoir brdfReservoir = RTXDI_SampleBrdf(rng, surface, sampleParams, params, brdfSample);

    DIReservoir state = EmptyDIReservoir();
//     RTXDI_CombineReservoirs(state, localReservoir, 0.5, localReservoir.targetPdf);
//     bool selectInfinite = RTXDI_CombineReservoirs(state, infiniteReservoir, RAB_GetNextRandom(rng), infiniteReservoir.targetPdf);
// #if RTXDI_ENABLE_PRESAMPLING
//     bool selectEnvironment = RTXDI_CombineReservoirs(state, environmentReservoir, RAB_GetNextRandom(rng), environmentReservoir.targetPdf);
// #endif
//     bool selectBrdf = RTXDI_CombineReservoirs(state, brdfReservoir, RAB_GetNextRandom(rng), brdfReservoir.targetPdf);
    
//     RTXDI_FinalizeResampling(state, 1.0, 1.0);
//     state.M = 1;

//     if (selectBrdf)
//         o_lightSample = brdfSample;
//     else
// #if RTXDI_ENABLE_PRESAMPLING
//     if (selectEnvironment)
//         o_lightSample = environmentSample;
//     else
// #endif
//     if (selectInfinite)
//         o_lightSample = infiniteSample;
//     else
//         o_lightSample = localSample;

    return state;
}

/** Initialize the sample parameters struct with default values.
* Defined so that so these can be compile time constants as defined by the user
* brdfCutoff Value in range [0,1] to determine how much to shorten BRDF rays. 0 to disable shortening
*/
ReSTIR_DI_SampleParameters ReSTIR_DI_InitSampleParameters(
    uint numRegirSamples,
    uint numLocalLightSamples,
    uint numInfiniteLightSamples,
    uint numEnvironmentMapSamples,
    uint numBrdfSamples,
    float brdfCutoff,
    float brdfRayMinT
) {
    ReSTIR_DI_SampleParameters result;
    result.numRegirSamples = numRegirSamples;
    result.numLocalLightSamples = numLocalLightSamples;
    result.numInfiniteLightSamples = numInfiniteLightSamples;
    result.numEnvironmentMapSamples = numEnvironmentMapSamples;
    result.numBrdfSamples = numBrdfSamples;
    result.numMisSamples = numLocalLightSamples + numEnvironmentMapSamples + numBrdfSamples;
    result.localLightMisWeight = float(numLocalLightSamples) / result.numMisSamples;
    result.environmentMapMisWeight = float(numEnvironmentMapSamples) / result.numMisSamples;
    result.brdfMisWeight = float(numBrdfSamples) / result.numMisSamples;
    result.brdfCutoff = brdfCutoff;
    result.brdfRayMinT = brdfRayMinT;
    return result;
}

#endif