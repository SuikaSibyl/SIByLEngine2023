#ifndef SRENDERER_ADDON_ASVGF_COMMON_HEADER_
#define SRENDERER_ADDON_ASVGF_COMMON_HEADER_

#include "../../include/common/cpp_compatible.hlsli"

/**
 * This addon implements the Adaptive Spatio-Temporal Variance-Guided Filtering (A-SVGF) algorithm.
 *
 * The Addon is based on the following paper:
 * 1. "Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for
 *  Path-Traced Global Illumination" by C.Schied et al.
 * @url: https://research.nvidia.com/publication/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A
 * 2. "Gradient Estimation for Real-Time Adaptive Temporal Filtering" by C.Schied et al.
 * @url: https://cg.ivd.kit.edu/atf.php
 *
 * The code is adapted from the following repository:
 * @url: https://github.com/NVIDIA/Q2RTX/blob/master/src/refresh/vkpt/shader/asvgf.glsl
 *
 * The copyright of the original code:
 * @copyright: /* Copyright (C) 2018 Christoph Schied
 * Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 * */

static const float wavelet_factor = 0.5;
static const float wavelet_kernel[2][2] = {
    { 1.0, wavelet_factor },
    { wavelet_factor, wavelet_factor *wavelet_factor }
};

#ifndef GRAD_DWN
#define GRAD_DWN 3
#endif // GRAD_DWN

#ifndef STRATUM_OFFSET_SHIFT
#define STRATUM_OFFSET_SHIFT 3
#endif // STRATUM_OFFSET_SHIFT

#define STRATUM_OFFSET_MASK ((1 << STRATUM_OFFSET_SHIFT) - 1)

/** Unpack the stratum offset from a packed offset. */
int2 UnpackStratumOffset(uint packed_offset) {
    return int2(int(packed_offset >> (STRATUM_OFFSET_SHIFT * 0)),
                int(packed_offset >> (STRATUM_OFFSET_SHIFT * 1))) & STRATUM_OFFSET_MASK;
}

/** Pack the stratum offset tp a packed offset. */
uint PackStratumOffset(in_ref(int2) found_offset) {
    return (1 << 7)                                          /* mark sample as busy */
           | (found_offset.x << (STRATUM_OFFSET_SHIFT * 0))  /* encode pos in */
           | (found_offset.y << (STRATUM_OFFSET_SHIFT * 1)); /* current frame */
}

#define STORAGE_SCALE_LF 1024
#define STORAGE_SCALE_HF 32
#define STORAGE_SCALE_SPEC 32
#define STORAGE_SCALE_HDR 128

#endif // SRENDERER_ADDON_ASVGF_COMMON_HEADER_