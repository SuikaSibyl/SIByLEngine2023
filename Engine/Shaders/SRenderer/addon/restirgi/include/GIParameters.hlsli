/***************************************************************************
 # Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#ifndef RESTIR_GI_PARAMETERS_H
#define RESTIR_GI_PARAMETERS_H

struct PackedGIReservoir {
    float3      position;
    uint32_t    packed_miscData_age_M; // See Reservoir.hlsli about the detail of the bit field.
    uint32_t    packed_radiance;    // Stored as 32bit LogLUV format.
    float       weight;
    uint32_t    packed_normal;      // Stored as 2x 16-bit snorms in the octahedral mapping
    float       unused;
};

#endif // RTXDI_PARAMETERS_H
