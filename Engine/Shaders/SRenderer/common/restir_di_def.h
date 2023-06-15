#ifndef _RESTIR_DIRECTIONAL_LIGHT_DEF_HEADER_
#define _RESTIR_DIRECTIONAL_LIGHT_DEF_HEADER_

/****************************************************
* This file contains the important definition 
* used for ReSTIR DI. Include this file in first,
* so that you could declare the data structure.
****************************************************/

struct DIReservoir {
    uint lightData;     // Light index (bits 0..30) and validity bit (31)
    uint uvData;        // Sample UV encoded in 16-bit fixed point format
    // Overloaded: represents RIS weight sum during streaming,
    // then reservoir weight (inverse PDF) after FinalizeResampling
    float weightSum;
    float targetPdf;    // Target PDF of the selected sample
    float M;            // Number of samples considered for this reservoir (pairwise MIS makes this a float)
    uint packedVisibility;  // Visibility information stored in the reservoir for reuse
    // Screen-space distance between the current location of the reservoir
    // and the location where the visibility information was generated,
    // minus the motion vectors applied in temporal resampling
    ivec2 spatialDistance; 
    uint age;               // How many frames ago the visibility information was generated
    float canonicalWeight;  // Cannonical weight when using pairwise MIS (ignored except during pairwise MIS computations)
};

/**
* Packed version of DIReservoir, used for storage in a structured buffer.
* Adding some clamping to the data to avoid overflows:
* # M is clamped to [0, 16383] (14 bits)
* # Age is clamped to [0, 255] (8 bits)
* # Spatial distance is clamped to [-127, 127] (8 bits each channel)
* # Visibility is clamped to [0, 262143] (18 bits)
*/
struct DIReservoirPacked {
    uint lightData;     // Light index (bits 0..30) and validity bit (31)
    uint uvData;        // Sample UV encoded in 16-bit fixed point format
    uint mVisibility;   // Low 18 bits: packed visibility, high 14 bits: M (up to 16383)
    uint distanceAge;   // Low 16 bits: spatial distance, high 8 bits: age, 8 bits: unused
    float targetPdf;    // Target PDF of the selected sample
    float weight;       // Reservoir weight (inverse PDF) ???
};

/**
* Represents a point on a light and its radiance,
* weighted relative to the surface that was used to generate the sample.
* Light samples are produced by the `SamplePolymorphicLight` function which takes a `DILightInfo`, 
* a `GBufferSurface`, and a pair of random numbers. 
* Internally, the instances of `DILightSample` are only used to compute the target PDF through 
* `GetLightSampleTargetPdfForSurface` and are not stored anywhere.
* Light samples that are stored and reused by ReSTIR are stored as sample references,
* or instances of `DILightSampleRef` structure that only stores the light index and the random numbers.
* Then the actual position on the lights are re-calculated for each surface they are weighed against.
*/
struct DILightSample {
    vec3  position;
    vec3  normal;
    vec3  radiance;
    float solidAnglePdf;
};

/**
* Stores information about a polymorphic light, i.e. a light of any type. 
* Typically, this structure would contain a field encoding the light type, another field storing the light radiance.
* , and other fields like position and orientation, whose interpretation depends on the specific light type.
* It's not a requirement however, and an implementation could choose to store lights of different types in different arrays,
* and keep only the light type and array index in the `RAB_LightInfo` structure, loading the specific light information
* only when sampling or weighing the light is performed.
*/
struct LightInfo {
    // uint4[0]
    vec3 center;
    uint scalars;    // 2x float16
    // uint4[1]
    uvec2 radiance;  // fp16x4
    uint direction1; // oct-encoded
    uint direction2; // oct-encoded
};


// Reservoirs are stored in a structured buffer in a block-linear layout.
// This constant defines the size of that block, measured in pixels.
#define RTXDI_RESERVOIR_BLOCK_SIZE 16

struct ReSTIR_DI_LocalLightRuntimeParameters {
    uint firstLocalLight;
    uint numLocalLights;
    uint enableLocalLightImportanceSampling;
    uint pad1;
};

struct ReSTIR_DI_InfiniteLightRuntimeParameters {
    uint firstInfiniteLight;
    uint numInfiniteLights;
    uint pad1;
    uint pad2;
};

struct ReSTIR_DI_EnvironmentLightRuntimeParameters {
    uint environmentLightPresent;
    uint environmentLightIndex;
    uint environmentRisBufferOffset;
    uint environmentTileSize;

    uint environmentTileCount;
    uint pad1;
    uint pad2;
    uint pad3;
};

struct ReSTIR_DI_RISBufferRuntimeParameters {
    uint tileSize;
    uint tileCount;
    uint pad1;
    uint pad2;
};

struct ReSTIR_DI_ResamplingRuntimeParameters {
    ReSTIR_DI_LocalLightRuntimeParameters localLightParams;
    ReSTIR_DI_InfiniteLightRuntimeParameters infiniteLightParams;
    ReSTIR_DI_EnvironmentLightRuntimeParameters environmentLightParams;
    ReSTIR_DI_RISBufferRuntimeParameters risBufferParams;
    
    uint neighborOffsetMask;
    uint uniformRandomNumber;
    uint activeCheckerboardField; // 0 - no checkerboard, 1 - odd pixels, 2 - even pixels
    uint reservoirBlockRowPitch;
    
    uint reservoirArrayPitch;
    uint pad1;
    uint pad2;
    uint pad3;

    uint firstLocalLight;
    uint numLocalLights;
};

struct ReSTIR_DI_SampleParameters {
    uint numRegirSamples;
    uint numLocalLightSamples;
    uint numInfiniteLightSamples;
    uint numEnvironmentMapSamples;
    uint numBrdfSamples;

    uint numMisSamples;
    float localLightMisWeight;
    float environmentMapMisWeight;
    float brdfMisWeight;
    float brdfCutoff; 
    float brdfRayMinT;
};

#endif