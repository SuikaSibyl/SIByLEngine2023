#include "../../../include/common/cpp_compatible.hlsli"
#include "../../../include/common/packing.hlsli"
#include "RuntimeParams.hlsli"

/**
 * This file is adapted from the original file in the RTXDI SDK.
 * @url: https://github.com/NVIDIAGameWorks/RTXDI/blob/main/rtxdi-sdk/include/rtxdi/GIReservoir.hlsli#L145
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

#ifndef GI_RESERVOIR_HLSLI
#define GI_RESERVOIR_HLSLI

// This structure represents a indirect lighting reservoir that stores the radiance and weight
// as well as its the position where the radiane come from.
struct GIReservoir {
    // postion of the 2nd bounce surface.
    float3 position;
    // normal vector of the 2nd bounce surface.
    float3 normal;
    // incoming radiance from the 2nd bounce surface.
    float3 radiance;
    // Overloaded: represents RIS weight sum during streaming,
    // then reservoir weight (inverse PDF) after FinalizeResampling
    float weightSum;
    // Number of samples considered for this reservoir
    uint M;
    // Number of frames the chosen sample has survived.
    uint age;
};

struct PackedGIReservoir {
    float3 position;
    uint32_t packed_miscData_age_M; // See Reservoir.hlsli about the detail of the bit field.
    uint32_t packed_radiance;       // Stored as 32bit LogLUV format.
    float weight;
    uint32_t packed_normal; // Stored as 2x 16-bit snorms in the octahedral mapping
    float unused;
};

// Encoding helper constants for RTXDI_PackedGIReservoir
static const uint RTXDI_PackedGIReservoir_MShift = 0;
static const uint RTXDI_PackedGIReservoir_MaxM = 0x0ff;

static const uint RTXDI_PackedGIReservoir_AgeShift = 8;
static const uint RTXDI_PackedGIReservoir_MaxAge = 0x0ff;

// "misc data" only exists in the packed form of GI reservoir and stored into a gap field of the packed form.
// RTXDI SDK doesn't look into this field at all and when it stores a packed GI reservoir, the field is always filled with zero.
// Application can use this field to store anything.
static const uint RTXDI_PackedGIReservoir_MiscDataMask = 0xffff0000;

// Converts a GIReservoir into its packed form.
// This function should be used only when the application needs to store data with the given argument.
// It can be retrieved when unpacking the GIReservoir, but RTXDI SDK doesn't use the filed at all.
PackedGIReservoir PackGIReservoir(in_ref(GIReservoir) reservoir, const uint miscData) {
    PackedGIReservoir data;
    data.position = reservoir.position;
    data.packed_normal = EncodeNormalizedVectorToSnorm2x16(reservoir.normal);
    data.packed_miscData_age_M =
        (miscData & RTXDI_PackedGIReservoir_MiscDataMask)
        | (min(reservoir.age, RTXDI_PackedGIReservoir_MaxAge) << RTXDI_PackedGIReservoir_AgeShift)
        | (min(reservoir.M, RTXDI_PackedGIReservoir_MaxM) << RTXDI_PackedGIReservoir_MShift);
    data.weight = reservoir.weightSum;
    data.packed_radiance = PackRGBE(reservoir.radiance);
    data.unused = 0;
    return data;
}

// Converts a PackedGIReservoir into its unpacked form.
// This function should be used only when the application wants to retrieve the misc data stored in the gap field of the packed form.
GIReservoir UnpackGIReservoir(PackedGIReservoir data, out uint miscData) {
    GIReservoir res;
    res.position = data.position;
    res.normal = DecodeNormalizedVectorFromSnorm2x16(data.packed_normal);
    res.radiance = UnpackRGBE(data.packed_radiance);
    res.weightSum = data.weight;
    res.M = (data.packed_miscData_age_M >> RTXDI_PackedGIReservoir_MShift) & RTXDI_PackedGIReservoir_MaxM;
    res.age = (data.packed_miscData_age_M >> RTXDI_PackedGIReservoir_AgeShift) & RTXDI_PackedGIReservoir_MaxAge;
    miscData = data.packed_miscData_age_M & RTXDI_PackedGIReservoir_MiscDataMask;
    return res;
}

/** Converts a PackedGIReservoir into its unpacked form. */
GIReservoir UnpackGIReservoir(PackedGIReservoir data) {
    uint miscFlags; // unused;
    return UnpackGIReservoir(data, miscFlags);
}

/** Create an empty GIReservoir */
GIReservoir EmptyGIReservoir() {
    GIReservoir s;
    s.position = float3(0.0, 0.0, 0.0);
    s.normal = float3(0.0, 0.0, 0.0);
    s.radiance = float3(0.0, 0.0, 0.0);
    s.weightSum = 0.0; s.M = 0; s.age = 0;
    return s;
}

// Creates a GI reservoir from a raw light sample.
// Note: the original sample PDF can be embedded into sampleRadiance, 
// in which case the samplePdf parameter should be set to 1.0.
GIReservoir MakeGIReservoir(
    in_ref(float3) samplePos,
    in_ref(float3) sampleNormal,
    in_ref(float3) sampleRadiance,
    in_ref(float) samplePdf
) {
    GIReservoir reservoir;
    reservoir.position = samplePos;
    reservoir.normal = sampleNormal;
    reservoir.radiance = sampleRadiance;
    reservoir.weightSum = samplePdf > 0.0 ? 1.0 / samplePdf : 0.0;
    reservoir.M = 1;
    reservoir.age = 0;
    return reservoir;
}

/** Tell whether a GIReservoir is valid */
bool IsValidGIReservoir(in_ref(GIReservoir) reservoir) {
    return reservoir.M != 0;
}

GIReservoir LoadGIReservoir(
    in_ref(GIResamplingRuntimeParameters) params,
    in_ref(uint2) reservoirPosition,
    in_ref(uint) reservoirArrayIndex,
    in_ref(RWStructuredBuffer<PackedGIReservoir>) reservoir_buffer,
) {
    const uint pointer = ReservoirPositionToPointer(params, reservoirPosition, reservoirArrayIndex);
    return UnpackGIReservoir(reservoir_buffer[pointer]);
}

GIReservoir LoadGIReservoir(
    in_ref(GIResamplingRuntimeParameters) params,
    in_ref(uint2) reservoirPosition,
    in_ref(uint) reservoirArrayIndex,
    in_ref(StructuredBuffer<PackedGIReservoir>) reservoir_buffer,
) {
    const uint pointer = ReservoirPositionToPointer(params, reservoirPosition, reservoirArrayIndex);
    return UnpackGIReservoir(reservoir_buffer[pointer]);
}

GIReservoir LoadGIReservoir(
    in_ref(GIResamplingRuntimeParameters) params,
    in_ref(uint2) reservoirPosition,
    in_ref(uint) reservoirArrayIndex,
    in_ref(RWStructuredBuffer<PackedGIReservoir>) reservoir_buffer,
    out_ref(uint) miscFlags
) {
    const uint pointer = ReservoirPositionToPointer(params, reservoirPosition, reservoirArrayIndex);
    return UnpackGIReservoir(reservoir_buffer[pointer], miscFlags);
}

GIReservoir LoadGIReservoir(
    in_ref(GIResamplingRuntimeParameters) params,
    in_ref(uint2) reservoirPosition,
    in_ref(uint) reservoirArrayIndex,
    in_ref(StructuredBuffer<PackedGIReservoir>) reservoir_buffer,
    out_ref(uint) miscFlags
) {
    const uint pointer = ReservoirPositionToPointer(params, reservoirPosition, reservoirArrayIndex);
    return UnpackGIReservoir(reservoir_buffer[pointer], miscFlags);
}

void StorePackedGIReservoir(
    in_ref(PackedGIReservoir) packedGIReservoir,
    in_ref(GIResamplingRuntimeParameters) params,
    in_ref(uint2) reservoirPosition,
    in_ref(uint) reservoirArrayIndex,
    in_ref(RWStructuredBuffer<PackedGIReservoir>) reservoir_buffer
) {
    const uint pointer = ReservoirPositionToPointer(params, reservoirPosition, reservoirArrayIndex);
    reservoir_buffer[pointer] = packedGIReservoir;
}

void StoreGIReservoir(
    in_ref(GIReservoir) reservoir,
    in_ref(GIResamplingRuntimeParameters) params,
    in_ref(uint2) reservoirPosition,
    in_ref(uint) reservoirArrayIndex,
    in_ref(RWStructuredBuffer<PackedGIReservoir>) reservoir_buffer
) {
    StorePackedGIReservoir(
        PackGIReservoir(reservoir, 0), params, 
        reservoirPosition, reservoirArrayIndex, reservoir_buffer);
}

void StoreGIReservoir(
    in_ref(GIReservoir) reservoir,
    in_ref(uint) miscFlags,
    in_ref(GIResamplingRuntimeParameters) params,
    in_ref(uint2) reservoirPosition,
    in_ref(uint) reservoirArrayIndex,
    in_ref(RWStructuredBuffer<PackedGIReservoir>) reservoir_buffer
) {
    StorePackedGIReservoir(
        PackGIReservoir(reservoir, miscFlags), params, 
        reservoirPosition, reservoirArrayIndex, reservoir_buffer);
}

#endif // GI_RESERVOIR_HLSLI
