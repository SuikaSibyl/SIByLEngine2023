/**
 * This file is (partially) adapted from NVIDIA's RTXGI SDK sample.
 * @url: https://github.com/NVIDIAGameWorks/RTXGI
 * @file: PathTraceRGS.hlsl
 * @copyright:
 * /*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 * /*
*/

#ifndef _SRENDERER_ADDON_DDGI_ROOTCONSTANTS_HEADER_
#define _SRENDERER_ADDON_DDGI_ROOTCONSTANTS_HEADER_

struct DDGIRootConstants {
    uint volumeIndex;
    uint volumeConstantsIndex;
    uint volumeResourceIndicesIndex;
    // Split uint3 into three uints to prevent internal padding
    // while keeping these values at the end of the struct
    uint reductionInputSizeX;
    uint reductionInputSizeY;
    uint reductionInputSizeZ;
    // Fucntions
    uint GetDDGIVolumeIndex() { return volumeIndex; }
    uint GetDDGIVolumeConstantsIndex() { return volumeConstantsIndex; }
    uint GetDDGIVolumeResourceIndicesIndex() { return volumeResourceIndicesIndex; }
    uint3 GetReductionInputSize() { return uint3(reductionInputSizeX, reductionInputSizeY, reductionInputSizeZ); }
};

#endif // !_SRENDERER_ADDON_DDGI_ROOTCONSTANTS_HEADER_