#ifndef _DDGI_ADDON_DDGI_VOLUME_DESC_GPU_H_
#define _DDGI_ADDON_DDGI_VOLUME_DESC_GPU_H_

/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
//------------------------------------------------------------------------
// Defines
//------------------------------------------------------------------------

// Bindless resource implementation type
#define RTXGI_BINDLESS_TYPE_RESOURCE_ARRAYS 0
#define RTXGI_BINDLESS_TYPE_DESCRIPTOR_HEAP 1

// Texture formats (matches EDDGIVolumeTextureFormat)
#define RTXGI_DDGI_VOLUME_TEXTURE_FORMAT_U32 0
#define RTXGI_DDGI_VOLUME_TEXTURE_FORMAT_F16 1
#define RTXGI_DDGI_VOLUME_TEXTURE_FORMAT_F16x2 2
#define RTXGI_DDGI_VOLUME_TEXTURE_FORMAT_F16x4 3
#define RTXGI_DDGI_VOLUME_TEXTURE_FORMAT_F32 4
#define RTXGI_DDGI_VOLUME_TEXTURE_FORMAT_F32x2 5
#define RTXGI_DDGI_VOLUME_TEXTURE_FORMAT_F32x4 6

// The number of fixed rays that are used by probe relocation and classification.
// These rays directions are always the same to produce temporally stable results.
#define RTXGI_DDGI_NUM_FIXED_RAYS 32

// Probe classification states
#define RTXGI_DDGI_PROBE_STATE_ACTIVE 0   // probe shoots rays and may be sampled by a front facing surface or another probe (recursive irradiance)
#define RTXGI_DDGI_PROBE_STATE_INACTIVE 1 // probe doesn't need to shoot rays, it isn't near a front facing surface

// Volume movement types
#define RTXGI_DDGI_VOLUME_MOVEMENT_TYPE_DEFAULT 0
#define RTXGI_DDGI_VOLUME_MOVEMENT_TYPE_SCROLLING 1

/**
 * Describes the location (i.e. index) of DDGIVolume resources
 * on the D3D descriptor heap or in bindless resource arrays.
 */
struct DDGIVolumeResourceIndices {
    uint rayDataUAVIndex;         // Index of the ray data UAV on the descriptor heap or in a RWTexture2D resource array
    uint rayDataSRVIndex;         // Index of the ray data SRV on the descriptor heap or in a Texture2D resource array
    uint probeIrradianceUAVIndex; // Index of the probe irradiance UAV on the descriptor heap or in a RWTexture2DArray resource array
    uint probeIrradianceSRVIndex; // Index of the probe irradiance SRV on the descriptor heap or in a Texture2DArray resource array
    //------------------------------------------------- 16B
    uint probeDistanceUAVIndex; // Index of the probe distance UAV on the descriptor heap or in a RWTexture2DArray resource array
    uint probeDistanceSRVIndex; // Index of the probe distance SRV on the descriptor heap or in a Texture2DArray resource array
    uint probeDataUAVIndex;     // Index of the probe data UAV on the descriptor heap or in a RWTexture2DArray resource array
    uint probeDataSRVIndex;     // Index of the probe data SRV on the descriptor heap or in a Texture2DArray resource array
    //------------------------------------------------- 32B
    uint probeVariabilityUAVIndex;        // Index of the probe variability UAV on the descriptor heap or in a RWTexture2DArray resource Array
    uint probeVariabilitySRVIndex;        // Index of the probe variability SRV on the descriptor heap or in a Texture2DArray resource array
    uint probeVariabilityAverageUAVIndex; // Index of the probe variability average UAV on the descriptor heap or in a RWTexture2DArray resource Array
    uint probeVariabilityAverageSRVIndex; // Index of the probe variability average SRV on the descriptor heap or in a Texture2DArray resource array
    //------------------------------------------------- 48B
};

/**
 * Describes the properties of a DDGIVolume, with values packed to compact formats.
 * This version of the struct uses 128B to store some values at full precision.
 */
struct DDGIVolumeDescGPUPacked {
    float3 origin;
    float probeHysteresis;
    //------------------------------------------------- 16B
    float4 rotation;
    //------------------------------------------------- 32B
    float4 probeRayRotation;
    //------------------------------------------------- 48B
    float probeMaxRayDistance;
    float probeNormalBias;
    float probeViewBias;
    float probeDistanceExponent;
    //------------------------------------------------- 64B
    float probeIrradianceEncodingGamma;
    float probeIrradianceThreshold;
    float probeBrightnessThreshold;
    float probeMinFrontfaceDistance;
    //------------------------------------------------- 80B
    float3 probeSpacing;
    uint packed0; // probeCounts.x (10), probeCounts.y (10), probeCounts.z (10), unused (2)
    //------------------------------------------------- 96B
    uint packed1; // probeRandomRayBackfaceThreshold (16), probeFixedRayBackfaceThreshold (16)
    uint packed2; // probeNumRays (16), probeNumIrradianceInteriorTexels (8), probeNumDistanceInteriorTexels (8)
    uint packed3; // probeScrollOffsets.x (15) sign bit (1), probeScrollOffsets.y (15) sign bit (1)
    uint packed4; // probeScrollOffsets.z (15) sign bit (1)
                  // movementType (1), probeRayDataFormat (3), probeIrradianceFormat (3), probeRelocationEnabled (1)
                  // probeClassificationEnabled (1), probeVariabilityEnabled (1)
                  // probeScrollClear Y-Z plane (1), probeScrollClear X-Z plane (1), probeScrollClear X-Y plane (1)
                  // probeScrollDirection Y-Z plane (1), probeScrollDirection X-Z plane (1), probeScrollDirection X-Y plane (1)
    //------------------------------------------------- 112B
    uint4 reserved; // 16B reserved for future use
    //------------------------------------------------- 128B
};

/**
 * Describes the properties of a DDGIVolume.
 */
struct DDGIVolumeDescGPU {
    float3 origin; // world-space location of the volume center

    float4 rotation;         // rotation quaternion for the volume
    float4 probeRayRotation; // rotation quaternion for probe rays

    uint movementType; // type of movement the volume allows. 0: default, 1: infinite scrolling

    float3 probeSpacing; // world-space distance between probes
    int3 probeCounts;    // number of probes on each axis of the volume

    int probeNumRays;                     // number of rays traced per probe
    int probeNumIrradianceInteriorTexels; // number of texels in one dimension of a probe's irradiance texture (does not include 1-texel border)
    int probeNumDistanceInteriorTexels;   // number of texels in one dimension of a probe's distance texture (does not include 1-texel border)

    float probeHysteresis;              // weight of the previous irradiance and distance data store in probes
    float probeMaxRayDistance;          // maximum world-space distance a probe ray can travel
    float probeNormalBias;              // offset along the surface normal, applied during lighting to avoid numerical instabilities when determining visibility
    float probeViewBias;                // offset along the camera view ray, applied during lighting to avoid numerical instabilities when determining visibility
    float probeDistanceExponent;        // exponent used during visibility testing. High values react rapidly to depth discontinuities, but may cause banding
    float probeIrradianceEncodingGamma; // exponent that perceptually encodes irradiance for faster light-to-dark convergence

    float probeIrradianceThreshold;        // threshold to identify when large lighting changes occur
    float probeBrightnessThreshold;        // threshold that specifies the maximum allowed difference in brightness between the previous and current irradiance values
    float probeRandomRayBackfaceThreshold; // threshold that specifies the ratio of *random* rays traced for a probe that may hit back facing triangles before the probe is considered inside geometry (used in blending)

    // Probe Relocation, Probe Classification
    float probeFixedRayBackfaceThreshold; // threshold that specifies the ratio of *fixed* rays traced for a probe that may hit back facing triangles before the probe is considered inside geometry (used in relocation & classification)
    float probeMinFrontfaceDistance;      // minimum world-space distance to a front facing triangle allowed before a probe is relocated

    // Infinite Scrolling Volumes
    int3 probeScrollOffsets;       // grid-space offsets used for scrolling movement
    bool probeScrollClear[3];      // whether probes of a plane need to be cleared due to scrolling movement
    bool probeScrollDirections[3]; // direction of scrolling movement (0: negative, 1: positive)

    // Feature Options
    uint probeRayDataFormat;         // texture format of the ray data texture (EDDGIVolumeTextureFormat)
    uint probeIrradianceFormat;      // texture format of the irradiance texture (EDDGIVolumeTextureFormat)
    bool probeRelocationEnabled;     // whether probe relocation is enabled for this volume
    bool probeClassificationEnabled; // whether probe classification is enabled for this volume
    bool probeVariabilityEnabled;    // whether probe variability is enabled for this volume
};

// Unpacks the compacted DDGIVolumeDescGPUPacked structure and returns the full sized structure.
DDGIVolumeDescGPU UnpackDDGIVolumeDescGPU(DDGIVolumeDescGPUPacked input) {
    DDGIVolumeDescGPU output = (DDGIVolumeDescGPU)0;
    output.origin = input.origin;
    output.probeHysteresis = input.probeHysteresis;
    output.rotation = input.rotation;
    output.probeRayRotation = input.probeRayRotation;
    output.probeMaxRayDistance = input.probeMaxRayDistance;
    output.probeNormalBias = input.probeNormalBias;
    output.probeViewBias = input.probeViewBias;
    output.probeDistanceExponent = input.probeDistanceExponent;
    output.probeIrradianceEncodingGamma = input.probeIrradianceEncodingGamma;
    output.probeIrradianceThreshold = input.probeIrradianceThreshold;
    output.probeBrightnessThreshold = input.probeBrightnessThreshold;
    output.probeMinFrontfaceDistance = input.probeMinFrontfaceDistance;
    output.probeSpacing = input.probeSpacing;

    // Probe Counts
    output.probeCounts.x = input.packed0 & 0x000003FF;
    output.probeCounts.y = (input.packed0 >> 10) & 0x000003FF;
    output.probeCounts.z = (input.packed0 >> 20) & 0x000003FF;

    // Thresholds
    output.probeRandomRayBackfaceThreshold = (float)(input.packed1 & 0x0000FFFF) / 65535.f;
    output.probeFixedRayBackfaceThreshold = (float)((input.packed1 >> 16) & 0x0000FFFF) / 65535.f;

    // Counts
    output.probeNumRays = input.packed2 & 0x0000FFFF;
    output.probeNumIrradianceInteriorTexels = (input.packed2 >> 16) & 0x000000FF;
    output.probeNumDistanceInteriorTexels = (input.packed2 >> 24) & 0x000000FF;

    // Probe Scroll Offsets
    output.probeScrollOffsets.x = input.packed3 & 0x00007FFF;
    if ((input.packed3 >> 15) & 0x00000001) output.probeScrollOffsets.x *= -1;
    output.probeScrollOffsets.y = (input.packed3 >> 16) & 0x00007FFF;
    if ((input.packed3 >> 31) & 0x00000001) output.probeScrollOffsets.y *= -1;
    output.probeScrollOffsets.z = (input.packed4) & 0x00007FFF;
    if ((input.packed4 >> 15) & 0x00000001) output.probeScrollOffsets.z *= -1;

    // Feature Bits
    output.movementType = (input.packed4 >> 16) & 0x00000001;
    output.probeRayDataFormat = (uint)((input.packed4 >> 17) & 0x00000007);
    output.probeIrradianceFormat = (uint)((input.packed4 >> 20) & 0x00000007);
    output.probeRelocationEnabled = (bool)((input.packed4 >> 23) & 0x00000001);
    output.probeClassificationEnabled = (bool)((input.packed4 >> 24) & 0x00000001);
    output.probeVariabilityEnabled = (bool)((input.packed4 >> 25) & 0x00000001);
    output.probeScrollClear[0] = (bool)((input.packed4 >> 26) & 0x00000001);
    output.probeScrollClear[1] = (bool)((input.packed4 >> 27) & 0x00000001);
    output.probeScrollClear[2] = (bool)((input.packed4 >> 28) & 0x00000001);
    output.probeScrollDirections[0] = (bool)((input.packed4 >> 29) & 0x00000001);
    output.probeScrollDirections[1] = (bool)((input.packed4 >> 30) & 0x00000001);
    output.probeScrollDirections[2] = (bool)((input.packed4 >> 31) & 0x00000001);

    return output;
}

// Returns true if the provided volume's infinite scrolling movement feature is enabled.
bool IsVolumeMovementScrolling(DDGIVolumeDescGPU volume) {
    return (volume.movementType == RTXGI_DDGI_VOLUME_MOVEMENT_TYPE_SCROLLING);
}

// RTXGI_PUSH_CONSTS_TYPE may be passed in as a define at shader compilation time.
// This define specifies how the shader will reference the push constants data block.
// If not using DDGI push constants, this define can be ignored.

#define RTXGI_PUSH_CONSTS_TYPE_SDK 1
#define RTXGI_PUSH_CONSTS_TYPE_APPLICATION 2

#if RTXGI_PUSH_CONSTS_TYPE == RTXGI_PUSH_CONSTS_TYPE_APPLICATION

// Note: Vulkan only allows a single block of memory for push constants. When using an
// application's pipeline layout in RTXGI shaders, the RTXGI shaders must understand
// the organization of the application's push constants data block!

// RTXGI_PUSH_CONSTS_VARIABLE_NAME must be passed in as a define at shader compilation time.
// This define specifies the variable name of the push constants block.
#ifndef RTXGI_PUSH_CONSTS_VARIABLE_NAME
#error Required define RTXGI_PUSH_CONSTS_VARIABLE_NAME is not defined!
#endif

// RTXGI_PUSH_CONSTS_FIELD_DDGI_VOLUME_INDEX_NAME must be passed in as a define at shader compilation time.
// This define specifies the name of the volume index field in the push constants struct.
#ifndef RTXGI_PUSH_CONSTS_FIELD_DDGI_VOLUME_INDEX_NAME
#error Required define RTXGI_PUSH_CONSTS_FIELD_DDGI_VOLUME_INDEX_NAME is not defined!
#endif

#if RTXGI_DECLARE_PUSH_CONSTS

// RTXGI_PUSH_CONSTS_STRUCT_NAME must be passed in as a define at shader compilation time.
// This define specifies the name of the push constants type struct.
#ifndef RTXGI_PUSH_CONSTS_STRUCT_NAME
#error Required define RTXGI_PUSH_CONSTS_STRUCT_NAME is not defined!
#endif

struct RTXGI_PUSH_CONSTS_STRUCT_NAME
{
    // IMPORTANT: insert padding to match the layout of your push constants!
    // The padding below matches the size of the Test Harness' "GlobalConstants" struct
    // with 48 float values before the DDGIRootConstants (see test-harness/include/graphics/Types.h)
    float4x4 padding0;
    float4x4 padding1;
    float4x4 padding2;
    uint RTXGI_PUSH_CONSTS_FIELD_DDGI_VOLUME_INDEX_NAME;
    uint2 ddgi_pad0;
    uint RTXGI_PUSH_CONSTS_FIELD_DDGI_REDUCTION_INPUT_SIZE_X_NAME;
    uint RTXGI_PUSH_CONSTS_FIELD_DDGI_REDUCTION_INPUT_SIZE_Y_NAME;
    uint RTXGI_PUSH_CONSTS_FIELD_DDGI_REDUCTION_INPUT_SIZE_Z_NAME;
    uint2 ddgi_pad1;
};
[[vk::push_constant]] RTXGI_PUSH_CONSTS_STRUCT_NAME RTXGI_PUSH_CONSTS_VARIABLE_NAME;
#endif

uint GetDDGIVolumeIndex() { return RTXGI_PUSH_CONSTS_VARIABLE_NAME.RTXGI_PUSH_CONSTS_FIELD_DDGI_VOLUME_INDEX_NAME; }
uint3 GetReductionInputSize()
{
    return uint3(RTXGI_PUSH_CONSTS_VARIABLE_NAME.RTXGI_PUSH_CONSTS_FIELD_DDGI_REDUCTION_INPUT_SIZE_X_NAME,
                 RTXGI_PUSH_CONSTS_VARIABLE_NAME.RTXGI_PUSH_CONSTS_FIELD_DDGI_REDUCTION_INPUT_SIZE_Y_NAME,
                 RTXGI_PUSH_CONSTS_VARIABLE_NAME.RTXGI_PUSH_CONSTS_FIELD_DDGI_REDUCTION_INPUT_SIZE_Z_NAME);
}

#elif RTXGI_PUSH_CONSTS_TYPE == RTXGI_PUSH_CONSTS_TYPE_SDK

[[vk::push_constant]] ConstantBuffer<DDGIRootConstants> DDGI;
uint GetDDGIVolumeIndex() { return DDGI.volumeIndex; }
uint3 GetReductionInputSize() { return uint3(DDGI.reductionInputSizeX, DDGI.reductionInputSizeY, DDGI.reductionInputSizeZ); }

#endif // RTXGI_PUSH_CONSTS_TYPE

// These functions are not relevant in Vulkan since descriptor heap style bindless is not available
uint GetDDGIVolumeConstantsIndex() { return 0; }
uint GetDDGIVolumeResourceIndicesIndex() { return 0; }

#endif