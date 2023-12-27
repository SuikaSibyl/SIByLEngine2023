#ifndef _SRENDERER_ADDON_VXGUIDING_ITNERFACE_HEADER_
#define _SRENDERER_ADDON_VXGUIDING_ITNERFACE_HEADER_

#include "../../../include/common/cpp_compatible.hlsli"
#include "../../../include/common/geometry.hlsli"
#include "../../../include/common/light_impl.hlsli"
#include "../../../raytracer/primitives/quad.hlsli"
#include "../../slc/include/slc_interface.hlsli"
#include "../../vxgi/include/conetrace_utils.hlsli"
#include "../tree/shared.hlsli"


float UnpackIrradiance(uint packed, float scalar = 65535.0f) {
    return float(packed) / scalar;
}

uint PackIrradiance(float unpacked, float scalar = 65535.0f) {
    return uint(unpacked * scalar);
}

AABB UnpackCompactAABB(
    in_ref(AABB) voxel_bound,
    in_ref(uint3) packedMin,
    in_ref(uint3) packedMax,
) {
    float3 bound_extend = voxel_bound.max - voxel_bound.min;
    AABB compactAABB;
    compactAABB.min = packedMin / float(uint(0xffffffff)) * bound_extend + voxel_bound.min;
    compactAABB.max = packedMax / float(uint(0xffffffff)) * bound_extend + voxel_bound.min;
    return compactAABB;
}

static uint VXGuider_MAX_CAPACITY = 131072;

/***********************************************************************
 * Voxel Guiding Utils
 ***********************************************************************
 * This file contains some utility functions for voxel guiding.
 ***********************************************************************/

/**
 * Sample a discrete probability distribution function (PDF) with 8 elements.
 * @param pdf The PDF to sample from with 8 elements.
 * @param rnd A random number in [0, 1).
 * @param pdf_value The value of the PDF at the sampled index.
 * @return The index of the sampled value.
 */
int SampleDiscretePDF8(float pdf[8], float rnd, out float pdf_value) {
    // Compute the cumulative distribution function (CDF)
    float cdf[8];
    float sum = 0.0;
    pdf_value = 0.0;
    for (int i = 0; i < 8; ++i) {
        if (pdf[i] > 0.0)
            sum += pdf[i];
        cdf[i] = sum;
    }
    // If the sum of the PDF is 0, return -1
    if (sum == 0.0) return -1;
    rnd *= sum;
    // Find the index corresponding to the sampled value
    for (int i = 0; i < 8; ++i) {
        if (rnd <= cdf[i]) {
            pdf_value = pdf[i] / sum;
            return i;
        }
    }
    // Default return value (shouldn't happen if probabilities sum up to 1)
    return -1;
}

/**
 * Given a sample which is sampled from a discrete probability
 * distribution function (PDF) with 8 elements.
 * Compute the probability of choosing the sample.
 * @param pdf The PDF to sample from with 8 elements.
 * @param sample The index of the sampled value.
 * @return The probability of choosing the sample.
 */
float SampleDiscretePDF8Pdf(float pdf[8], int sample) {
    // Compute the cumulative distribution function (CDF)
    float sum = 0.0;
    for (int i = 0; i < 8; ++i) {
        if (pdf[i] > 0.0)
            sum += pdf[i];
    }
    // If the sum of the PDF is 0, return -1
    if (sum == 0.0) return -1;
    return pdf[sample] / sum;
}

/**
 * Create a spherical quad from a square quad.
 * @param local The local coordinate of the reference point position.
 * @param extend The extend of the square quad.
 * @return The spherical quad.
 */
SphQuad CreateSphQuad(in_ref(float3) local, float extend) {
    const float extend2 = extend + extend;
    SphQuad squad;
    SphQuadInit(float3(-extend, -extend, 0), float3(extend2, 0, 0), float3(0, extend2, 0), local, squad);
    return squad;
}

SphQuad CreateSphQuad(in_ref(float3) local, float2 extend) {
    const float2 extend2 = extend + extend;
    SphQuad squad;
    SphQuadInit(float3(-extend, 0), float3(extend2.x, 0, 0), float3(0, extend2.y, 0), local, squad);
    // if (local.z < 0) squad.S = 0;
    return squad;
}

/**
 * Sample a voxel as directional sampling.
 * @param center The center of the voxel.
 * @param extend The (half) extend of the voxel.
 * @param ref_point The reference point position.
 * @param rnds The random numbers.
 * @param pdf The probability density function (PDF) value.
 * @return The sampled direction.
 */
float3 SampleSphericalVoxel(
    in_ref(float3) center,
    in_ref(float) extend,
    in_ref(float3) ref_point,
    in_ref(float3) rnds,
    out_ref(float) pdf)
{
    // sign: return -1 if x is less than zero; 0 if x equals zero; and 1 if x is greater than zero.
    const float3 dir_sign = sign(ref_point - center);
    const float3 x_center = center + float3(dir_sign.x * extend, 0, 0);
    const float3 y_center = center + float3(0, dir_sign.y * extend, 0);
    const float3 z_center = center + float3(0, 0, dir_sign.z * extend);

    const float3x3 rotations[3] = {
        float3x3(+0, +1, +0, +0, +0, +1, +1, +0, +0), // x axis rotation
        float3x3(+1, +0, +0, +0, +0, +1, +0, +1, +0), // y axis rotation
        float3x3(+1, +0, +0, +0, +1, +0, +0, +0, +1), // z axis rotation
    };

    float3 locals[3];
    SphQuad squads[3];
    locals[0] = mul(rotations[0] * dir_sign.x, ref_point - x_center);
    squads[0] = CreateSphQuad(locals[0], extend);
    locals[1] = mul(rotations[1] * dir_sign.y, ref_point - y_center);
    squads[1] = CreateSphQuad(locals[1], extend);
    locals[2] = mul(rotations[2] * dir_sign.z, ref_point - z_center);
    squads[2] = CreateSphQuad(locals[2], extend);

    float pdfs[3];
    pdfs[0] = (dir_sign.x == 0) ? 0 : squads[0].S;
    pdfs[1] = (dir_sign.y == 0) ? 0 : squads[1].S;
    pdfs[2] = (dir_sign.z == 0) ? 0 : squads[2].S;

    float cdfs[3];
    float sum = 0.0;
    pdf = 0.0;
    for (int i = 0; i < 3; ++i) {
        if (pdfs[i] > 0.0)
            sum += pdfs[i];
        cdfs[i] = sum;
    }
    // If the sum of the PDF is 0, return -1
    if (sum == 0.0) return float3(0.f / 0.f);
    rnds.x *= sum;

    // Find the index corresponding to the sampled value
    int select_face = -1;
    float select_face_pdf = 0.f;
    for (int i = 0; i < 3; ++i) {
        if (rnds.x <= cdfs[i]) {
            select_face = i;
            select_face_pdf = pdfs[i] / sum;
            break;
        }
    }

    float3 local_dir;
    float face_pdf;
    SampleSphQuad(locals[select_face], squads[select_face], rnds.yz, local_dir, face_pdf);

    pdf = select_face_pdf * face_pdf;
    return mul(local_dir, rotations[select_face] * dir_sign[select_face]);
}

float3 SampleSphericalVoxel(
    in_ref(float3) center,
    in_ref(float3) extend,
    in_ref(float3) ref_point,
    in_ref(float3) rnds,
    out_ref(float) pdf)
{
    // sign: return -1 if x is less than zero; 0 if x equals zero; and 1 if x is greater than zero.
    const float3 dir_sign = sign(ref_point - center);
    const float3 x_center = center + float3(dir_sign.x * extend.x, 0, 0);
    const float3 y_center = center + float3(0, dir_sign.y * extend.y, 0);
    const float3 z_center = center + float3(0, 0, dir_sign.z * extend.z);

    const float3x3 rotations[3] = {
        float3x3(+0, +1, +0, +0, +0, +1, +1, +0, +0), // x axis rotation
        float3x3(+1, +0, +0, +0, +0, +1, +0, +1, +0), // y axis rotation
        float3x3(+1, +0, +0, +0, +1, +0, +0, +0, +1), // z axis rotation
    };

    float3 locals[3];
    SphQuad squads[3];
    locals[0] = mul(rotations[0] * dir_sign.x, ref_point - x_center);
    squads[0] = CreateSphQuad(locals[0], extend.yz);
    locals[1] = mul(rotations[1] * dir_sign.y, ref_point - y_center);
    squads[1] = CreateSphQuad(locals[1], extend.xz);
    locals[2] = mul(rotations[2] * dir_sign.z, ref_point - z_center);
    squads[2] = CreateSphQuad(locals[2], extend.xy);

    float pdfs[3];
    pdfs[0] = (dir_sign.x == 0) || isnan(squads[0].S) ? 0 : squads[0].S;
    pdfs[1] = (dir_sign.y == 0) || isnan(squads[1].S) ? 0 : squads[1].S;
    pdfs[2] = (dir_sign.z == 0) || isnan(squads[2].S) ? 0 : squads[2].S;

    float cdfs[3];
    pdf = 0.0;
    float sum = 0.0;
    for (int i = 0; i < 3; ++i) {
        if (pdfs[i] > 0.0)
            sum += pdfs[i];
        cdfs[i] = sum;
    }
    // If the sum of the PDF is 0, return -1
    if (sum == 0.0) return float3(0.f / 0.f);
    rnds.x *= sum;

    // Find the index corresponding to the sampled value
    int select_face = -1;
    float select_face_pdf = 0.f;
    for (int i = 0; i < 3; ++i) {
        if (rnds.x <= cdfs[i]) {
            select_face = i;
            select_face_pdf = pdfs[i] / sum;
            break;
        }
    }

    float3 local_dir;
    float face_pdf;
    SampleSphQuad(locals[select_face], squads[select_face], rnds.yz, local_dir, face_pdf);

    pdf = select_face_pdf * face_pdf;
    return mul(local_dir, rotations[select_face] * dir_sign[select_face]);
}

float PdfSampleSphericalVoxel(
    in_ref(float3) center,
    in_ref(float3) extend,
    in_ref(float3) ref_point
) {
    // sign: return -1 if x is less than zero; 0 if x equals zero; and 1 if x is greater than zero.
    const float3 dir_sign = sign(ref_point - center);
    const float3 x_center = center + float3(dir_sign.x * extend.x, 0, 0);
    const float3 y_center = center + float3(0, dir_sign.y * extend.y, 0);
    const float3 z_center = center + float3(0, 0, dir_sign.z * extend.z);

    const float3x3 rotations[3] = {
        float3x3(+0, +1, +0, +0, +0, +1, +1, +0, +0), // x axis rotation
        float3x3(+1, +0, +0, +0, +0, +1, +0, +1, +0), // y axis rotation
        float3x3(+1, +0, +0, +0, +1, +0, +0, +0, +1), // z axis rotation
    };

    float3 locals[3];
    SphQuad squads[3];
    locals[0] = mul(rotations[0] * dir_sign.x, ref_point - x_center);
    squads[0] = CreateSphQuad(locals[0], extend.yz);
    locals[1] = mul(rotations[1] * dir_sign.y, ref_point - y_center);
    squads[1] = CreateSphQuad(locals[1], extend.xz);
    locals[2] = mul(rotations[2] * dir_sign.z, ref_point - z_center);
    squads[2] = CreateSphQuad(locals[2], extend.xy);

    float pdfs[3];
    pdfs[0] = (dir_sign.x == 0) || isnan(squads[0].S) ? 0 : squads[0].S;
    pdfs[1] = (dir_sign.y == 0) || isnan(squads[1].S) ? 0 : squads[1].S;
    pdfs[2] = (dir_sign.z == 0) || isnan(squads[2].S) ? 0 : squads[2].S;
    
    return 1.f / (pdfs[0] + pdfs[1] + pdfs[2]);
}

void NormalizePDF8(inout_ref(float) weight[8]) {
    float sum = 0.0;
    for (int i = 0; i < 8; ++i)
        sum += weight[i];
    for (int i = 0; i < 8; ++i)
        weight[i] = (sum == 0) ? 0.125f : weight[i] / sum;
}

// void SLCDistanceFactor(
//     inout_ref(float) weights[8],
//     in_ref(float) l2_min[8],
//     in_ref(float) l2_max[8],
//     SLCDistanceMode mode
// ) {
//     if (mode == SLCDistanceMode::kAverageMinmax) {
//         // if all elements of l2_min are zeros, just ignore them
//         float w_max0[8];
//         float w_min0[8];
//         float weight_sum = 0.0;
//         int zero_count = 0;
//         for (int i = 0; i < 8; ++i)
//             if (l2_min[i] == 0) ++zero_count;
//         float sum_w_max0 = 0.0;
//         float sum_w_min0 = 0.0;
//         for (int i = 0; i < 8; ++i) {
//             if (zero_count == 8)
//                 w_min0[i] = weights[i];
//             else if (zero_count > 0)
//                 w_max0[i] = (l2_min[i] == 0) ? weights[i] : 0;
//             else
//                 w_max0[i] = weights[i] / l2_min[i];
//             w_min0[i] = weights[i] / l2_max[i];
//             sum_w_max0 += w_max0[i];
//             sum_w_min0 += w_min0[i];
//             weight_sum += weights[i];
//         }
//         for (int i = 0; i < 8; ++i) {
//             w_max0[i] = (sum_w_max0 == 0) ? (weights[i] / weight_sum) : (w_max0[i] / sum_w_max0);
//             w_min0[i] /= sum_w_min0;
//             weights[i] = (w_max0[i] + w_min0[i]) * 0.5f;
//         }
//         return;
//     }
//     return;
// }

int3 SampleByLuminance(
    Texture3D<float2> tex[6],
    in_ref(float3) ref_point,
    in_ref(int) maxMipLevel,
    in_ref(VoxelTexInfo) info,
    inout_ref(RandomSamplerState) RNG,
    out_ref(float) pdf
) {
    pdf = 1.f;
    int3 selectedVox = int3(0);
    for (int mipLevel = maxMipLevel; mipLevel >= 0; --mipLevel) {
        selectedVox *= 2;
        float vox_pdfs[8];
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) {
                    int3 coord = int3(i, j, k) + selectedVox;
                    AABB voxel_bound = VoxelToBound(coord, mipLevel, info);
                    const float3 direction = normalize((voxel_bound.max + voxel_bound.min) * 0.5 - ref_point);
                    float2 radopa = LoadRadopa(tex, coord, mipLevel, direction);
                    vox_pdfs[i * 4 + j * 2 + k] = radopa.x;
                }
        float vox_pdf;
        const int vox = SampleDiscretePDF8(vox_pdfs, GetNextRandom(RNG), vox_pdf);
        if (vox == -1) return int3(-1);
        int3 selectedVoxLocal = int3((vox >> 2) & 1, (vox >> 1) & 1, vox & 1);
        selectedVox += selectedVoxLocal;
        pdf *= vox_pdf;
    }
    return selectedVox;
}

int3 SampleByLuminanceOcclusion(
    Texture3D<float2> tex[6],
    in_ref(SamplerState) sampler,
    in_ref(float3) ref_point,
    in_ref(float3) ref_normal,
    in_ref(int) maxMipLevel,
    in_ref(VoxelTexInfo) info,
    inout_ref(RandomSamplerState) RNG,
    out_ref(float) pdf
) {
    pdf = 1.f;
    int3 selectedVox = int3(0);
    for (int mipLevel = maxMipLevel; mipLevel >= 0; --mipLevel) {
        selectedVox *= 2;
        float vox_pdfs[8];
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) {
                    int3 coord = int3(i, j, k) + selectedVox;
                    AABB voxel_bound = VoxelToBound(coord, mipLevel, info);
                    const float3 direction = normalize((voxel_bound.max + voxel_bound.min) * 0.5 - ref_point);
                    float2 radopa = LoadRadopa(tex, coord, mipLevel, direction);
                    const float occlusion = mipLevel == 5 ?
                        TraceConeOcclusionToVoxel(tex, sampler, ref_point, ref_normal, info, voxel_bound, mipLevel, 50, 1.0f) : 1.f;
                    vox_pdfs[i * 4 + j * 2 + k] = radopa.x * occlusion;
                }
        float vox_pdf;
        const int vox = SampleDiscretePDF8(vox_pdfs, GetNextRandom(RNG), vox_pdf);
        if (vox == -1) return int3(-1);
        int3 selectedVoxLocal = int3((vox >> 2) & 1, (vox >> 1) & 1, vox & 1);
        selectedVox += selectedVoxLocal;
        pdf *= vox_pdf;
    }
    return selectedVox;
}

/**
 * Trace cone into voxel grid.
 * @param tex 6 faces of anisotropic voxel representation
 * @param sampler sampler state
 * @param position start position
 * @param normal normal at start position, only used to avoid self collision
 * @param direction cone direction
 * @param tan_half_aperture cone aperture
 * @param info voxel texture info
 * @param voxelBound voxel grid bound
 * @param voxMipLevel voxel mip level
 * @param maxDistWS max distance in world space
 * @param samplingFactor sampling factor
 */
float TraceConeOcclusionToVoxel(
    in_ref(Texture3D<VOXEL_TYPE>) tex[6],
    in_ref(SamplerState) sampler,
    in_ref(float3) position,
    in_ref(float3) normal,
    in_ref(VoxelTexInfo) info,
    in_ref(AABB) voxelBound,
    in_ref(int) voxMipLevel,
    in_ref(float) maxDistWS,
    in_ref(float) samplingFactor = 1.0f
) {
    float3 direction = (voxelBound.max + voxelBound.min) * 0.5 - position;
    const float dir_dist = length(direction);
    direction = direction / dir_dist;
    const float local_radius = distance(voxelBound.max, voxelBound.min) * 0.81649658 * 0.5;
    const float tan_half_aperture = min(local_radius / dir_dist, 1.0);
    // Compute visible face indices
    const int3 visibleFace = ComputeVoxelFaceIndices(direction);
    // world space grid voxel size
    float voxelWorldSize = 1.0 / (info.voxelScale * info.volumeDimension);
    // weight per axis for aniso sampling
    const float3 weight = direction * direction;
    // move further to avoid self collision
    float dist = voxelWorldSize; // move 1 voxel size to avoid self collision
    const float3 startPosition = position + normal * dist * 2;
    // final results
    float coneSample = float(0.0f);
    float occlusion = 0.0f;
    float maxDistance = maxDistWS / info.voxelScale;
    // out of boundaries check
    float enter = 0.0;
    float leave = 0.0;
    // ray marching loop
    while (coneSample < 1.0f && dist <= maxDistance) {
        const float3 conePosition = startPosition + direction * dist;
        // cone expansion and respective mip level based on diameter
        const float diameter = 2.0f * tan_half_aperture * dist;
        const float mipLevel = log2(diameter / voxelWorldSize);
        if (int(mipLevel) >= voxMipLevel) break;
        // convert position to texture coord
        const float3 coord = WorldToVoxel(conePosition, info);
        // get directional sample from anisotropic representation
        float anisoSample = AnistropicSampleOcclusion(tex, sampler, coord, weight, visibleFace, mipLevel);
        // front to back composition, also accumulate occlusion
        coneSample += (1.0f - coneSample) * anisoSample;
        // move further into volume
        dist += diameter * samplingFactor;
    }
    return coneSample;
}

float SampleByLuminancePdf(
    Texture3D<float2> tex[6],
    in_ref(float3) ref_point,
    in_ref(int) maxMipLevel,
    in_ref(int4) sample,
    in_ref(VoxelTexInfo) info
) {
    float pdf = 1.f;
    int3 selectedVox = int3(0);
    for (int mipLevel = maxMipLevel; mipLevel >= sample.w; --mipLevel) {
        selectedVox *= 2;
        float vox_pdfs[8];
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) {
                    int3 coord = int3(i, j, k) + selectedVox;
                    AABB voxel_bound = VoxelToBound(coord, mipLevel, info);
                    const float3 direction = normalize((voxel_bound.max + voxel_bound.min) * 0.5 - ref_point);
                    float2 radopa = LoadRadopa(tex, coord, mipLevel, direction);
                    vox_pdfs[i * 4 + j * 2 + k] = radopa.x;
                }
        const int mip_offset = sample.w - mipLevel;
        int3 sampledVox = mip_offset >= 0 ? (sample.xyz << mip_offset) : (sample.xyz >> -mip_offset);
        sampledVox -= selectedVox;
        const int vox = sampledVox.x * 4 + sampledVox.y * 2 + sampledVox.z;
        float vox_pdf = SampleDiscretePDF8Pdf(vox_pdfs, vox);
        int3 selectedVoxLocal = int3((vox >> 2) & 1, (vox >> 1) & 1, vox & 1);
        selectedVox = sampledVox;
        pdf *= vox_pdf;
    }
    return pdf;
}

float SampleByLuminanceOcclusionPdf(
    Texture3D<float2> tex[6],
    SamplerState sampler,
    in_ref(float3) ref_point,
    in_ref(float3) ref_normal,
    in_ref(int) maxMipLevel,
    in_ref(int4) sample,
    in_ref(VoxelTexInfo) info
) {
    float pdf = 1.f;
    int3 selectedVox = int3(0);
    for (int mipLevel = maxMipLevel; mipLevel >= sample.w; --mipLevel) {
        selectedVox *= 2;
        float vox_pdfs[8];
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) {
                    int3 coord = int3(i, j, k) + selectedVox;
                    AABB voxel_bound = VoxelToBound(coord, mipLevel, info);
                    const float3 direction = normalize((voxel_bound.max + voxel_bound.min) * 0.5 - ref_point);
                    float2 radopa = LoadRadopa(tex, coord, mipLevel, direction);
                    const float occlusion = TraceConeOcclusionToVoxel(tex, sampler, ref_point, ref_normal, info, voxel_bound, mipLevel, 50, 1.0f);
                    vox_pdfs[i * 4 + j * 2 + k] = radopa.x * occlusion;
                }
        const int mip_offset = sample.w - mipLevel;
        int3 sampledVox = mip_offset >= 0 ? (sample.xyz << mip_offset) : (sample.xyz >> -mip_offset);
        sampledVox -= selectedVox;
        const int vox = sampledVox.x * 4 + sampledVox.y * 2 + sampledVox.z;
        float vox_pdf = SampleDiscretePDF8Pdf(vox_pdfs, vox);
        int3 selectedVoxLocal = int3((vox >> 2) & 1, (vox >> 1) & 1, vox & 1);
        selectedVox = sampledVox;
        pdf *= vox_pdf;
    }
    return pdf;
}

struct VXGuidingSetting {
    bool considerDistanceTerm;
};

int3 SampleByEstimation(
    Texture3D<float2> tex[6],
    SamplerState sampler,
    in_ref(float3) ref_point,
    in_ref(float3) ref_normal,
    in_ref(int) maxMipLevel,
    in_ref(VoxelTexInfo) info,
    inout_ref(RandomSamplerState) RNG,
    out_ref(float) pdf,
    in_ref(VXGuidingSetting) setting
) {
    // GeometryTermSetting geom_setting;
    // geom_setting.useApproximateCosineBound = false;

    pdf = 1.f;
    // int3 selectedVox = int3(0);
    // for (int mipLevel = maxMipLevel; mipLevel >= 0; --mipLevel) {
    //     selectedVox *= 2;
    //     float vox_pdfs[8];
    //     float l2_min[8];
    //     float l2_max[8];
    //     for (int i = 0; i < 2; ++i)
    //         for (int j = 0; j < 2; ++j)
    //             for (int k = 0; k < 2; ++k) {
    //                 int3 coord = int3(i, j, k) + selectedVox;
    //                 AABB voxel_bound = VoxelToBound(coord, mipLevel, info);
    //                 const float3 direction = normalize((voxel_bound.max + voxel_bound.min) * 0.5 - ref_point);
    //                 float2 radopa = LoadRadopa(tex, coord, mipLevel, direction);
    //                 float weight = radopa.x; // light
    //                 weight *= ComputeGeometryTerm(ref_point, ref_normal, voxel_bound, geom_setting);
    //                 const int index = i * 4 + j * 2 + k;
    //                 l2_min[index] = SquaredDistanceToClosestPoint(ref_point, voxel_bound);
    //                 l2_max[index] = SquaredDistanceToFarthestPoint(ref_point, voxel_bound);
    //                 vox_pdfs[index] = weight;
    //             }
    //     if (setting.considerDistanceTerm)
    //         SLCDistanceFactor(vox_pdfs, l2_min, l2_max, SLCDistanceMode::kAverageMinmax);
    //     float vox_pdf;
    //     const int vox = SampleDiscretePDF8(vox_pdfs, GetNextRandom(RNG), vox_pdf);
    //     if (vox == -1) return int3(-1);
    //     int3 selectedVoxLocal = int3((vox >> 2) & 1, (vox >> 1) & 1, vox & 1);
    //     selectedVox += selectedVoxLocal;
    //     pdf *= vox_pdf;
    // }
    return int3(0);
}

int3 SampleByEstimationTwoPass(
    Texture3D<float2> tex[6],
    SamplerState sampler,
    in_ref(float3) ref_point,
    in_ref(float3) ref_normal,
    in_ref(int) maxMipLevel,
    in_ref(VoxelTexInfo) info,
    inout_ref(RandomSamplerState) RNG,
    out_ref(float) pdf,
    in_ref(VXGuidingSetting) setting
) {
    pdf = 0.f;
    // GeometryTermSetting geom_setting;
    // geom_setting.useApproximateCosineBound = false;

    // pdf = 1.f;
    // int4 selectedVox = int4(0, 0, 0, maxMipLevel + 1);
    // // In the first pass, we select one voxel hierarchically using WRS
    // int3 voxelRefpointIn = int3(0);
    // int3 voxelRefpointInNext = int3(0);
    // float weightSum = 0.0;
    // float weightSelected = 0.0;

    // for (int mipLevel = maxMipLevel; mipLevel >= 0; --mipLevel) {
    //     voxelRefpointIn.xyz *= 2;
    //     float vox_pdfs[8];
    //     float l2_min[8];
    //     float l2_max[8];
    //     for (int i = 0; i < 2; ++i)
    //         for (int j = 0; j < 2; ++j)
    //             for (int k = 0; k < 2; ++k) {
    //                 int3 coord = int3(i, j, k) + voxelRefpointIn;
    //                 const float3 direction = normalize((coord + float3(0.5)) - ref_point);
    //                 float2 radopa = LoadRadopa(tex, coord, mipLevel, direction);
    //                 float weight = radopa.x; // light
    //                 AABB voxel_bound = VoxelToBound(coord, mipLevel, info);
    //                 weight *= ComputeGeometryTerm(ref_point, ref_normal, voxel_bound, geom_setting);
    //                 const int index = i * 4 + j * 2 + k;
    //                 const float minDist = SquaredDistanceToClosestPoint(ref_point, voxel_bound);
    //                 const float maxDist = SquaredDistanceToFarthestPoint(ref_point, voxel_bound);
    //                 if (setting.considerDistanceTerm) {
    //                     weight /= (minDist + maxDist) * 0.5f;
    //                 }
    //                 // If the refpoint is in the voxel, we set it as voxelRefpointInNext
    //                 if (all(voxel_bound.min < ref_point) && all(ref_point < voxel_bound.max)) {
    //                     voxelRefpointInNext = coord;
    //                     weight = 0; // and we set the weight to 0
    //                 }
    //                 // Do WRS
    //                 weightSum += weight;
    //                 if (GetNextRandom(RNG) < weight / weightSum) {
    //                     selectedVox = int4(coord, mipLevel);
    //                     weightSelected = weight;
    //                 }
    //             }
    //     voxelRefpointIn = voxelRefpointInNext;
    // }
    // // pdf of selecting the voxel from the first pass
    // pdf = weightSelected / weightSum;

    // // In the second pass, we select one voxel hierarchically using WRS
    // for (int mipLevel = selectedVox.w - 1; mipLevel >= 0; --mipLevel) {
    //     selectedVox.xyz *= 2;
    //     float vox_pdfs[8];
    //     float l2_min[8];
    //     float l2_max[8];
    //     for (int i = 0; i < 2; ++i)
    //         for (int j = 0; j < 2; ++j)
    //             for (int k = 0; k < 2; ++k) {
    //                 int3 coord = int3(i, j, k) + selectedVox.xyz;
    //                 AABB voxel_bound = VoxelToBound(coord, mipLevel, info);
    //                 const float3 direction = normalize((voxel_bound.max + voxel_bound.min) * 0.5 - ref_point);
    //                 float2 radopa = LoadRadopa(tex, coord, mipLevel, direction);
    //                 float weight = radopa.x; // light
    //                 weight *= ComputeGeometryTerm(ref_point, ref_normal, voxel_bound, geom_setting);
    //                 const int index = i * 4 + j * 2 + k;
    //                 l2_min[index] = SquaredDistanceToClosestPoint(ref_point, voxel_bound);
    //                 l2_max[index] = SquaredDistanceToFarthestPoint(ref_point, voxel_bound);
    //                 vox_pdfs[index] = weight;
    //             }
    //     if (setting.considerDistanceTerm)
    //         SLCDistanceFactor(vox_pdfs, l2_min, l2_max, SLCDistanceMode::kAverageMinmax);
    //     float vox_pdf;
    //     const int vox = SampleDiscretePDF8(vox_pdfs, GetNextRandom(RNG), vox_pdf);
    //     if (vox == -1) return int3(-1);
    //     int3 selectedVoxLocal = int3((vox >> 2) & 1, (vox >> 1) & 1, vox & 1);
    //     selectedVox.xyz += selectedVoxLocal;
    //     pdf *= vox_pdf;
    // }
    return int3(0);
}

struct TreeEvaluateConfig {
    bool intensity_only;
    bool useApproximateCosineBound;
    int distanceType;
    // BSDF parameters
    bool bsdf_enabled;
    float Kd;
    float Ks;
    float eta;
    float roughness;
    float3x3 frame;

    __init() {
        intensity_only = false;
        useApproximateCosineBound = false;
        distanceType = 0;
        bsdf_enabled = false;
        Kd = 0;
        Ks = 0;
        eta = 0;
        roughness = 0;
        frame = float3x3(0);
    }
};

/**
 * Evaluate the first child weight in a binary tree.
 * @param c0 The first child node.
 * @param c1 The second child node.
 * @param p The reference point position.
 * @param n The reference point normal.
 * @param v The reference point view direction.
 * @param config The evaluation configuration.
 * @param prob0 The output probability of the first child.
 * @return True if the evaluation is successful.
 */
bool EvaluateFirstChildWeight(
    in_ref(TreeNode) c0,
    in_ref(TreeNode) c1,
    in_ref(float3) p,
    in_ref(float3) n,
    in_ref(float3) v,
    in_ref(TreeEvaluateConfig) config,
    out_ref(float) prob0
) {
    const float c0_intensity = c0.intensity;
    const float c1_intensity = c1.intensity;

    prob0 = 0.f;
    if (config.intensity_only) {
        if (c0_intensity == 0) {
            if (c1_intensity == 0) return false;
            prob0 = 0; return true;
        }
        else if (c1_intensity == 0) {
            prob0 = 1; return true;
        }

        prob0 = c0_intensity / (c0_intensity + c1_intensity);
        return true;
    }

    const float3 c0_boundMin = c0.getAABBMin();
    const float3 c0_boundMax = c0.getAABBMax();
    const float3 c1_boundMin = c1.getAABBMin();
    const float3 c1_boundMax = c1.getAABBMax();

    // Compute the weights
    float geom0 = 1;
    float geom1 = 1;

    if (config.useApproximateCosineBound) {
        geom0 = GeomTermBoundApproximate(p, n, c0_boundMin, c0_boundMax);
        geom1 = GeomTermBoundApproximate(p, n, c1_boundMin, c1_boundMax);
    }
    else {
        geom0 = GeomTermBound(p, n, c0_boundMin, c0_boundMax);
        geom1 = GeomTermBound(p, n, c1_boundMin, c1_boundMax);
    }

    // if (config.useLightCone) {
    //     const float3 c0r_boundMin = 2 * p - c0_boundMax;
    //     const float3 c0r_boundMax = 2 * p - c0_boundMin;
    //     const float3 c1r_boundMin = 2 * p - c1_boundMax;
    //     const float3 c1r_boundMax = 2 * p - c1_boundMin;

    //     float cos0 = 1;
    //     float cos1 = 1;

    //     if (config.useApproximateCosineBound) {
    //         cos0 = GeomTermBoundApproximate(p, c0.cone.xyz, c0r_boundMin, c0r_boundMax);
    //         cos1 = GeomTermBoundApproximate(p, c1.cone.xyz, c1r_boundMin, c1r_boundMax);
    //     }
    //     else {
    //         cos0 = GeomTermBound(p, c0.cone.xyz, c0r_boundMin, c0r_boundMax);
    //         cos1 = GeomTermBound(p, c1.cone.xyz, c1r_boundMin, c1r_boundMax);
    //     }

    //     geom0 *= max(0.f, cos(max(0.f, acos(cos0) - c0.cone.w)));
    //     geom1 *= max(0.f, cos(max(0.f, acos(cos1) - c1.cone.w)));
    // }

    if (config.bsdf_enabled) {
        AABB c0_bound; c0_bound.min = c0_boundMin; c0_bound.max = c0_boundMax;
        AABB c1_bound; c1_bound.min = c1_boundMin; c1_bound.max = c1_boundMax;
        geom0 *= ApproxBSDFWeight(p, config.frame, v, c0_bound, config.Kd, config.Ks, config.eta, config.roughness);
        geom1 *= ApproxBSDFWeight(p, config.frame, v, c1_bound, config.Kd, config.Ks, config.eta, config.roughness);
    }

    if (geom0 + geom1 == 0)
        return false;

    if (geom0 == 0) {
        prob0 = 0;
        return true;
    }
    else if (geom1 == 0) {
        prob0 = 1;
        return true;
    }

    const float intensGeom0 = c0_intensity * geom0;
    const float intensGeom1 = c1_intensity * geom1;

    float l2_min0;
    float l2_min1;
    l2_min0 = SquaredDistanceToClosestPoint(p, c0_boundMin, c0_boundMax);
    l2_min1 = SquaredDistanceToClosestPoint(p, c1_boundMin, c1_boundMax);

    if (config.distanceType == 0) {
        if (l2_min0 < WidthSquared(c0_boundMin, c0_boundMax) 
         || l2_min1 < WidthSquared(c1_boundMin, c1_boundMax)) {
            prob0 = intensGeom0 / (intensGeom0 + intensGeom1);
        }
        else {
            float w_max0 = normalizedWeights(l2_min0, l2_min1, intensGeom0, intensGeom1);
            prob0 = w_max0; // closest point
        }
    }
    else if (config.distanceType == 1) {
        const float3 l0 = 0.5 * (c0_boundMin + c0_boundMax) - p;
        const float3 l1 = 0.5 * (c1_boundMin + c1_boundMax) - p;
        const float w_max0 = normalizedWeights(max(0.001, dot(l0, l0)), max(0.001, dot(l1, l1)), intensGeom0, intensGeom1);
        prob0 = w_max0; // closest point
    }
    else if (config.distanceType == 2) {
        // avg weight of minmax (used in the paper)
        const float l2_max0 = SquaredDistanceToFarthestPoint(p, c0_boundMin, c0_boundMax);
        const float l2_max1 = SquaredDistanceToFarthestPoint(p, c1_boundMin, c1_boundMax);
        const float w_max0 = l2_min0 == 0 && l2_min1 == 0 ? intensGeom0 / (intensGeom0 + intensGeom1) : normalizedWeights(l2_min0, l2_min1, intensGeom0, intensGeom1);
        const float w_min0 = normalizedWeights(l2_max0, l2_max1, intensGeom0, intensGeom1);
        prob0 = 0.5 * (w_max0 + w_min0);
    }
    if (config.distanceType == 3) {
        prob0 = intensGeom0 / (intensGeom0 + intensGeom1);
    }
    return true;
}

/**
 * Traverse the light tree to pick a light.
 * @param nid The node ID to start the traversal from.
 * @param LeafStartIndex The index of the first leaf node.
 * @param p The reference point position.
 * @param n The reference point normal.
 * @param v The reference point view direction.
 * @param rnd The random number in [0, 1).
 * @param nprob The normalized probability of the picked light.
 * @param config The evaluation configuration.
 * @param nodeBuffer The light tree node buffer.
 */
inline int TraverseLightTree(
    int nid,
    int LeafStartIndex,
    in_ref(float3) p,
    in_ref(float3) n,
    in_ref(float3) v,
    in_ref(float) rnd,
    inout_ref(double) nprob,
    in_ref(TreeEvaluateConfig) config,
    StructuredBuffer<TreeNode> nodeBuffer
) {
    nprob = 1.;
    TreeNode node = nodeBuffer[nid];
    while (nid < LeafStartIndex) {
        uint16_t c0_id = node.left_idx; // left child
        uint16_t c1_id = node.right_idx; // right child

        float prob0;
        const TreeNode c0 = nodeBuffer[c0_id];
        const TreeNode c1 = nodeBuffer[c1_id];
        if (EvaluateFirstChildWeight(c0, c1, p, n, v, config, prob0)) {
            if (rnd < prob0) {
                nid = c0_id;
                rnd /= prob0;
                nprob *= double(prob0);
                node = c0;
            }
            else {
                nid = c1_id;
                rnd = (rnd - prob0) / (1 - prob0);
                nprob *= double(1 - prob0);
                node = c1;
            }
        }
        else {
            // dead branch and thus invalid sample
            nid = -1;
            break;
        }
    }
    return nid;
}

/**
 * Get the pdf of traversing the light tree to pick a light.
 * @param nid The node ID to start the traversal from.
 * @param LeafStartIndex The index of the first leaf node.
 * @param p The reference point position.
 * @param n The reference point normal.
 * @param v The reference point view direction.
 * @param rnd The random number in [0, 1).
 * @param nprob The normalized probability of the picked light.
 * @param config The evaluation configuration.
 * @param nodeBuffer The light tree node buffer.
 */
inline double PdfTraverseLightTree(
    int nid,
    int leafSelected,
    in_ref(float3) p,
    in_ref(float3) n,
    in_ref(float3) v,
    in_ref(TreeEvaluateConfig) config,
    StructuredBuffer<TreeNode> nodeBuffer,
) {
    double nprob = 1.;
    int node_index = leafSelected;
    TreeNode node = nodeBuffer[node_index];
    // Immediately return if it is the root node already
    if (node_index == 0 || node_index == nid) return nprob;

    while (node.parent_idx != 0xFFFFFFFF) {
        TreeNode parent = nodeBuffer[node.parent_idx];
        uint16_t c0_id = parent.left_idx;  // left child
        uint16_t c1_id = parent.right_idx; // right child

        float prob0;
        const TreeNode c0 = nodeBuffer[c0_id];
        const TreeNode c1 = nodeBuffer[c1_id];
        EvaluateFirstChildWeight(c0, c1, p, n, v, config, prob0);
        if (c0_id == node_index) {
            nprob *= double(prob0);
        } else {
            nprob *= double(1 - prob0);
        }
        node_index = node.parent_idx;
        node = parent;
        if (node_index == 0 || node_index == nid) return nprob;
    }
    return nprob;
}

inline double PdfTraverseLightTree_Intensity(
    int nid,
    int selectedIndex,
    StructuredBuffer<TreeNode> nodeBuffer
) {
    const float parent_intensity = nodeBuffer[nid].intensity;
    const float child_intensity = nodeBuffer[selectedIndex].intensity;
    return double(child_intensity) / double(parent_intensity);
}

int SampleTopLevelTree(
    in_ref(StructuredBuffer<float>) top_level_tree,
    in_ref(int) spixelID,
    in_ref(float) rnd,
    out_ref(double) pdf
) {
    const int topLevelOffset = spixelID * 64;
    int topIndex = 1;
    double top_pdf = 1.f;
    pdf = 0.f;
    const float topImportance = top_level_tree[topLevelOffset + topIndex];
    // If the top level node is dead, return -1.
    if (topImportance == 0.f) {
        pdf = 0.f;
        return -1;
    }
    // Else, sample the top level tree.
    for (int i = 0; i < 5; ++i) {
        const int leftIndex = topIndex << 1;
        const int rightIndex = leftIndex + 1;
        const float leftImportance = top_level_tree[topLevelOffset + leftIndex];
        const float rightImportance = top_level_tree[topLevelOffset + rightIndex];
        float prob0;
        if (leftImportance == 0) prob0 = 0.f;
        else if (rightImportance == 0) prob0 = 1.f;
        else prob0 = leftImportance / (leftImportance + rightImportance);
        
        if (rnd < prob0) {
            topIndex = leftIndex;
            rnd /= prob0;
            top_pdf *= double(prob0);
        }
        else {
            topIndex = rightIndex;
            rnd = (rnd - prob0) / (1 - prob0);
            top_pdf *= double(1 - prob0);
        }
    }
    topIndex -= 32;
    pdf = top_pdf;
    return topIndex;
}

float PdfSampleTopLevelTree(
    in_ref(StructuredBuffer<float>) top_level_tree,
    in_ref(int) spixelID,
    in_ref(int) topIndex
) {
    const int topLevelOffset = spixelID * 64;
    const float topImportance = top_level_tree[topLevelOffset + 1];
    const float leafImportance = top_level_tree[topLevelOffset + 32 + topIndex];
    return (topImportance == 0.f) ? 0.f : leafImportance / topImportance;
}

float3 EvaluateVPLIndirectLight(
    in_ref(int2) pixel,
    in_ref(Ray) primaryRay,
    in_ref(GeometryHit) primaryHit,
    in_ref(float4) bsdfPos,
    in_ref(RWTexture2D<float4>) color
) {
    const float4 bsdfColor = color[pixel];
    if (any(bsdfPos != 0) && bsdfColor.w != 0) {
        const float3 bsdf_direction = normalize(bsdfPos.xyz - primaryHit.position);
        const float bsdf_pdf = PdfBsdfSample(primaryHit, -primaryRay.direction, bsdf_direction);
        const float3 first_bsdf = EvalBsdf(primaryHit, -primaryRay.direction, bsdf_direction);
        if (bsdf_pdf <= 0) return float3(0);
        float3 throughput = first_bsdf / bsdf_pdf;
        return bsdfColor.xyz * throughput;
    }
    else {
        return float3(0, 0, 0);
    }
}

float3 EvaluateVPLIndirectLight(
    in_ref(int2) pixel,
    in_ref(Ray) primaryRay,
    in_ref(GeometryHit) primaryHit,
    in_ref(float4) bsdfPos,
    in_ref(RWTexture2D<float4>) color,
    out_ref(float) bsdf_pdf,
    out_ref(float3) throughput,
    out_ref(Ray) bsdfRay,
) {
    bsdf_pdf = 0;
    throughput = float3(0);
    bsdfRay.direction = float3(0);
    bsdfRay.origin = float3(0);
    const float4 bsdfColor = color[pixel];
    if (any(bsdfPos != 0) && bsdfColor.w != 0) {
        float3 bsdf_direction = normalize(bsdfPos.xyz - primaryHit.position);
        bsdfRay = SpawnRay(primaryHit, bsdf_direction);
        bsdf_pdf = PdfBsdfSample(primaryHit, -primaryRay.direction, bsdf_direction);
        float3 first_bsdf = EvalBsdf(primaryHit, -primaryRay.direction, bsdf_direction);
        if (bsdf_pdf <= 0) return float3(0);
        throughput = first_bsdf / bsdf_pdf;
        return bsdfColor.xyz * throughput;
    }
    else {
        return float3(0, 0, 0);
    }
}

float3 EvaluateIndirectLight(
    in_ref(Ray) primaryRay,
    in_ref(Ray) secondRay,
    double pdf,
    in_ref(GeometryHit) primaryHit,
    inout_ref(PrimaryPayload) payload,
    inout_ref(RandomSamplerState) RNG
) {
    if (pdf <= 0) return float3(0);
    // trace the second ray
    Intersection(secondRay, SceneBVH, payload, RNG);
    const float3 first_bsdf = EvalBsdf(primaryHit, -primaryRay.direction, secondRay.direction);
    const float3 throughput = first_bsdf; // divide float(pdf). leave it to return line;
    if (HasHit(payload.hit)) {
        const PolymorphicLightInfo light = lights[0];
        const float3 di = EvaluateDirectLight(secondRay, payload.hit, light, RNG);
        return di * throughput / float(pdf);
    }
    else return float3(0, 0, 0);
}

float3 EvaluateIndirectLightEX(
    in_ref(Ray) primaryRay,
    in_ref(Ray) secondRay,
    double pdf,
    in_ref(GeometryHit) primaryHit,
    inout_ref(PrimaryPayload) payload,
    inout_ref(RandomSamplerState) RNG,
    out_ref(float3) di
) {
    di = float3(0);
    if (pdf <= 0) return float3(0);
    // trace the second ray
    Intersection(secondRay, SceneBVH, payload, RNG);
    const float3 first_bsdf = EvalBsdf(primaryHit, -primaryRay.direction, secondRay.direction);
    const float3 throughput = first_bsdf; // divide float(pdf). leave it to return line;
    if (HasHit(payload.hit)) {
        const PolymorphicLightInfo light = lights[0];
        di = EvaluateDirectLight(secondRay, payload.hit, light, RNG);
        return di * throughput / float(pdf);
    }
    else return float3(0, 0, 0);
}

float3 EvaluateIndirectLightEX(
    in_ref(Ray) primaryRay,
    in_ref(Ray) secondRay,
    double pdf,
    in_ref(ShadingSurface) surface,
    inout_ref(PrimaryPayload) payload,
    inout_ref(RandomSamplerState) RNG,
    out_ref(float3) di
) {
    di = float3(0);
    if (pdf <= 0) return float3(0);
    // trace the second ray
    Intersection(secondRay, SceneBVH, payload, RNG);
    const float3 first_bsdf = EvalBsdf(surface, -primaryRay.direction, secondRay.direction);
    const float3 throughput = first_bsdf; // divide float(pdf). leave it to return line;
    if (HasHit(payload.hit)) {
        const PolymorphicLightInfo light = lights[0];
        di = EvaluateDirectLight(secondRay, payload.hit, light, RNG);
        return di * throughput / float(pdf);
    }
    else return float3(0, 0, 0);
}

SplitShading EvaluateIndirectLightEXSplit(
    in_ref(Ray) primaryRay,
    in_ref(Ray) secondRay,
    double pdf,
    in_ref(ShadingSurface) surface,
    inout_ref(PrimaryPayload) payload,
    inout_ref(RandomSamplerState) RNG,
    out_ref(float3) di
) {
    di = float3(0);
    SplitShading first_bsdf;
    first_bsdf.diffuse = float3(0);
    first_bsdf.specular = float3(0);
    if (pdf <= 0) return first_bsdf;
    // trace the second ray
    Intersection(secondRay, SceneBVH, payload, RNG);
    first_bsdf = EvalBsdfSplit(surface, -primaryRay.direction, secondRay.direction);
    // divide float(pdf). leave it to return line;
    if (HasHit(payload.hit)) {
        const PolymorphicLightInfo light = lights[0];
        di = EvaluateDirectLight(secondRay, payload.hit, light, RNG);
        first_bsdf.diffuse *= di / float(pdf);
        first_bsdf.specular *= di / float(pdf);
        return first_bsdf;
    }
    else return first_bsdf;
}

float3 EvaluateIndirectLightEXX(
    in_ref(Ray) primaryRay,
    in_ref(Ray) secondRay,
    double pdf,
    in_ref(GeometryHit) primaryHit,
    inout_ref(PrimaryPayload) payload,
    inout_ref(RandomSamplerState) RNG,
    out_ref(float3) di,
    out_ref(float3) throughput
) {
    di = float3(0);
    throughput = float3(0);
    if (pdf <= 0) return float3(0);
    // trace the second ray
    Intersection(secondRay, SceneBVH, payload, RNG);
    const float3 first_bsdf = EvalBsdf(primaryHit, -primaryRay.direction, secondRay.direction);
    const float3 ithroughput = first_bsdf; // divide float(pdf). leave it to return line;
    if (HasHit(payload.hit)) {
        const PolymorphicLightInfo light = lights[0];
        di = EvaluateDirectLight(secondRay, payload.hit, light, RNG);
        throughput = ithroughput / float(pdf);
        return di * throughput;
    }
    else return float3(0, 0, 0);
}

float3 EvaluateIndirectLightEXX(
    in_ref(Ray) primaryRay,
    in_ref(Ray) secondRay,
    double pdf,
    in_ref(ShadingSurface) surface,
    inout_ref(PrimaryPayload) payload,
    inout_ref(RandomSamplerState) RNG,
    out_ref(float3) di,
    out_ref(float3) throughput
) {
    di = float3(0);
    throughput = float3(0);
    if (pdf <= 0) return float3(0);
    // trace the second ray
    Intersection(secondRay, SceneBVH, payload, RNG);
    const float3 first_bsdf = EvalBsdf(surface, -primaryRay.direction, secondRay.direction);
    const float3 ithroughput = first_bsdf; // divide float(pdf). leave it to return line;
    if (HasHit(payload.hit)) {
        const PolymorphicLightInfo light = lights[0];
        di = EvaluateDirectLight(secondRay, payload.hit, light, RNG);
        throughput = ithroughput / float(pdf);
        return di * throughput;
    }
    else return float3(0, 0, 0);
}

float3 EvaluateIndirectLightEXXSplit(
    in_ref(Ray) primaryRay,
    in_ref(Ray) secondRay,
    double pdf,
    in_ref(ShadingSurface) surface,
    inout_ref(PrimaryPayload) payload,
    inout_ref(RandomSamplerState) RNG,
    out_ref(float3) di,
    out_ref(SplitShading) throughput
) {
    di = float3(0);
    throughput.diffuse = float3(0);
    throughput.specular = float3(0);
    if (pdf <= 0) return float3(0);
    // trace the second ray
    Intersection(secondRay, SceneBVH, payload, RNG);
    const SplitShading first_bsdf = EvalBsdfSplit(surface, -primaryRay.direction, secondRay.direction);
    if (HasHit(payload.hit)) {
        const PolymorphicLightInfo light = lights[0];
        di = EvaluateDirectLight(secondRay, payload.hit, light, RNG);
        throughput.diffuse = first_bsdf.diffuse / float(pdf);
        throughput.specular = first_bsdf.specular / float(pdf);
        return di;
    }
    else return float3(0, 0, 0);
}

float3 EvaluateIndirectLight_NoThroughput(
    in_ref(Ray) primaryRay,
    in_ref(Ray) secondRay,
    double pdf,
    in_ref(GeometryHit) primaryHit,
    inout_ref(PrimaryPayload) payload,
    inout_ref(RandomSamplerState) RNG,
    out_ref(float3) di
) {
    di = float3(0);
    if (pdf <= 0) return float3(0);
    // trace the second ray
    Intersection(secondRay, SceneBVH, payload, RNG);
    if (HasHit(payload.hit)) {
        const PolymorphicLightInfo light = lights[0];
        di = EvaluateDirectLight(secondRay, payload.hit, light, RNG);
        return di;
    }
    else return float3(0, 0, 0);
}

float3 SampleSphericalVoxel(
    in_ref(int) vxFlatten,
    in_ref(float3) rnd_vec,
    in_ref(GeometryHit) primaryHit,
    in_ref(StructuredBuffer<uint4>) pMin,
    in_ref(StructuredBuffer<uint4>) pMax,
    in_ref(VoxelTexInfo) info,
    out_ref(AABB) voxelBound,
    out_ref(float) pdf
) {
    const int3 vxID = ReconstructIndex(vxFlatten, 64);
    voxelBound = VoxelToBound(vxID, 0, info);
    const AABB compact_bound = UnpackCompactAABB(voxelBound, pMin[vxFlatten].xyz, pMax[vxFlatten].xyz);
    const float3 voxelExtent = float3(compact_bound.max - compact_bound.min) / 2;
    const float3 voxelCenter = 0.5 * (compact_bound.min + compact_bound.max);
    return SampleSphericalVoxel(
        voxelCenter,
        voxelExtent,
        primaryHit.position,
        rnd_vec,
        pdf);
}

float3 SampleSphericalVoxel(
    in_ref(int) vxFlatten,
    in_ref(float3) rnd_vec,
    in_ref(ShadingSurface) surface,
    in_ref(StructuredBuffer<uint4>) pMin,
    in_ref(StructuredBuffer<uint4>) pMax,
    in_ref(VoxelTexInfo) info,
    out_ref(AABB) voxelBound,
    out_ref(float) pdf
) {
    const int3 vxID = ReconstructIndex(vxFlatten, 64);
    voxelBound = VoxelToBound(vxID, 0, info);
    const AABB compact_bound = UnpackCompactAABB(voxelBound, pMin[vxFlatten].xyz, pMax[vxFlatten].xyz);
    const float3 voxelExtent = float3(compact_bound.max - compact_bound.min) / 2;
    const float3 voxelCenter = 0.5 * (compact_bound.min + compact_bound.max);
    return SampleSphericalVoxel(
        voxelCenter,
        voxelExtent,
        surface.worldPos,
        rnd_vec,
        pdf);
}

float PdfSampleSphericalVoxel(
    in_ref(int3) vxID,
    in_ref(int) vxFlatten,
    in_ref(GeometryHit) primaryHit,
    in_ref(StructuredBuffer<uint4>) pMin,
    in_ref(StructuredBuffer<uint4>) pMax,
    in_ref(VoxelTexInfo) info
) {
    const AABB aabb = VoxelToBound(vxID, 0, info);
    const AABB compact_bound = UnpackCompactAABB(aabb, pMin[vxFlatten].xyz, pMax[vxFlatten].xyz);
    const float3 voxelExtent = float3(compact_bound.max - compact_bound.min) / 2;
    const float3 voxelCenter = 0.5 * (compact_bound.min + compact_bound.max);
    return PdfSampleSphericalVoxel(
        voxelCenter,
        voxelExtent,
        primaryHit.position);
}

float PdfSampleSphericalVoxel(
    in_ref(int3) vxID,
    in_ref(int) vxFlatten,
    in_ref(ShadingSurface) surface,
    in_ref(StructuredBuffer<uint4>) pMin,
    in_ref(StructuredBuffer<uint4>) pMax,
    in_ref(VoxelTexInfo) info
) {
    const AABB aabb = VoxelToBound(vxID, 0, info);
    const AABB compact_bound = UnpackCompactAABB(aabb, pMin[vxFlatten].xyz, pMax[vxFlatten].xyz);
    const float3 voxelExtent = float3(compact_bound.max - compact_bound.min) / 2;
    const float3 voxelCenter = 0.5 * (compact_bound.min + compact_bound.max);
    return PdfSampleSphericalVoxel(
        voxelCenter,
        voxelExtent,
        surface.worldPos);
}

enum VoxelGuidingType {
    VG_Uniform,
    VG_Irradiance,
    VG_VisibilityIrradiance,
    VG_SLC
};

enum VisibilityType {
    None,
    Spixel,
    FuzzySpixel
};

struct VoxelGuidingConfig {
    VoxelGuidingType type;
    VisibilityType visibility;
};

float3 SampleVoxelGuiding(
    int VXCount,
    in_ref(int2) pixel,
    in_ref(float2) uv,
    in_ref(int3) currVXID,
    in_ref(GeometryHit) hit,
    in_ref(VoxelTexInfo) info,
    in_ref(VoxelGuidingConfig) config,
    inout_ref(RandomSamplerState) RNG,
    in_ref(StructuredBuffer<uint4>) p_min,
    in_ref(StructuredBuffer<uint4>) p_max,
    in_ref(StructuredBuffer<uint>) compact_indices,
    in_ref(StructuredBuffer<TreeNode>) tree_nodes,
    in_ref(const Texture2D<int>) spixel_idx,
    in_ref(const Texture2D<float4>) fuzzy_weight,
    in_ref(const Texture2D<int4>) fuzzyIDX,
    in_ref(const StructuredBuffer<float>) topLevelTree,
    in_ref(const StructuredBuffer<int>) clusterRoots,
    out_ref(AABB) aabb,
    out_ref(double) pdf
) {
    aabb.min = float3(+k_inf);
    aabb.max = float3(-k_inf);
    pdf = 0.f;
    int vxFlatten = -1;
    int3 vxID = currVXID;
    if (config.type == VoxelGuidingType::VG_Uniform) {
        const int selectedID = clamp(int(VXCount * GetNextRandom(RNG)), 0, VXCount - 1);
        vxFlatten = compact_indices[selectedID];
        vxID = ReconstructIndex(vxFlatten, 64);
        pdf = 1. / VXCount;
    }
    else if (config.type == VoxelGuidingType::VG_Irradiance) {
        TreeEvaluateConfig teconfig;
        teconfig.intensity_only = true;
        int selectedID = TraverseLightTree(
            0, VXCount - 1, hit.position, 
            hit.geometryNormal, float3(0), 
            GetNextRandom(RNG), pdf, teconfig, tree_nodes);
        if (selectedID != -1) {
            selectedID = tree_nodes[selectedID].vx_idx;
            vxFlatten = compact_indices[selectedID];
            vxID = ReconstructIndex(vxFlatten, 64);
        }
    }
    else if (config.type == VoxelGuidingType::VG_VisibilityIrradiance) {
        int spixelID = spixel_idx[pixel];
        if (config.visibility == VisibilityType::FuzzySpixel) {
            float4 weights = fuzzy_weight[pixel];
            int4 indices = fuzzyIDX[pixel];
            float fuzzy_rnd = GetNextRandom(RNG);
            float accum = 0.f;
            for (int i = 0; i < 4; ++i) {
                accum += weights[i];
                if (fuzzy_rnd < accum) {
                    spixelID = indices[i];
                    break;
                }
            }
        }

        // sample top level tree
        double top_pdf = 1.f;
        const int topIndex = SampleTopLevelTree(topLevelTree, spixelID, GetNextRandom(RNG), top_pdf);
        if (topIndex==-1) {
            pdf = 0.f;
            return float3(0);
        }

        TreeEvaluateConfig teconfig;
        const int clusterRoot = clusterRoots[topIndex];
        double bottom_pdf;
        int selectedID = TraverseLightTree(
            clusterRoot, VXCount - 1,
            hit.position, hit.geometryNormal, float3(0), GetNextRandom(RNG), bottom_pdf, teconfig, tree_nodes);
        if (selectedID != -1) {
            int tmp = selectedID;
            selectedID = tree_nodes[selectedID].vx_idx;
            pdf = top_pdf * bottom_pdf;
            vxFlatten = compact_indices[selectedID];
            vxID = ReconstructIndex(vxFlatten, 64);
        }
    }
    else if (config.type == VoxelGuidingType::VG_SLC) {
        TreeEvaluateConfig teconfig;
        teconfig.intensity_only = false;
        teconfig.distanceType = 0;
        teconfig.useApproximateCosineBound = true;
        int selectedID = TraverseLightTree(
            0, VXCount - 1, hit.position,
            hit.geometryNormal, float3(0),
            GetNextRandom(RNG), pdf, teconfig, tree_nodes);
        if (selectedID != -1) {
            selectedID = tree_nodes[selectedID].vx_idx;
            vxFlatten = compact_indices[selectedID];
            vxID = ReconstructIndex(vxFlatten, 64);
        }
    }
    // If the current voxel is the selected voxel, return 0.0
    if (all(vxID == currVXID)) {
        pdf = 0.0;
        return float3(0);
    }
    // Sample the spherical voxel
    float sph_pdf;
    float3 guidedDir = SampleSphericalVoxel(
        vxFlatten,
        float3(GetNextRandom(RNG), uv),
        hit,
        p_min,
        p_max,
        info,
        aabb,
        sph_pdf);
    pdf *= double(sph_pdf);
    return guidedDir;
}

double PdfVoxelGuiding(
    int VXCount,
    in_ref(int3) vxID,
    in_ref(int3) currVXID,
    in_ref(GeometryHit) hit,
    in_ref(VoxelTexInfo) info,
    in_ref(VoxelGuidingConfig) config,
    in_ref(StructuredBuffer<uint4>) p_min,
    in_ref(StructuredBuffer<uint4>) p_max,
    in_ref(StructuredBuffer<uint>) compact_indices,
    in_ref(StructuredBuffer<TreeNode>) tree_nodes,
    in_ref(const Texture3D<int>) inverse_index,
    in_ref(const StructuredBuffer<int>) compact2leaf,
    in_ref(Texture3D<uint>) vx_irradiance
) {
    double voxel_pdf = 0.0;
    const int vxFlatten = FlatIndex(vxID, info.volumeDimension);

    if (all(vxID == currVXID)) return 0.0;
    if (config.type == VoxelGuidingType::VG_Uniform) {
        const uint vxIrradiance = vx_irradiance[vxID];
        if (vxIrradiance == 0) return 0.0;
        voxel_pdf = 1. / VXCount;
    }
    else if (config.type == VoxelGuidingType::VG_Irradiance) {
        const int compactID = inverse_index[vxID];
        if (compactID == -1) return 0.0;
        const int leafID = compact2leaf[compactID];
        voxel_pdf = PdfTraverseLightTree_Intensity(0, leafID, tree_nodes);
    }
    else if (config.type == VoxelGuidingType::VG_SLC) {
        TreeEvaluateConfig teconfig;
        teconfig.intensity_only = false;
        teconfig.distanceType = 0;
        teconfig.useApproximateCosineBound = true;
        const int compactID = inverse_index[vxID];
        if (compactID == -1) return 0.0;
        const int leafID = compact2leaf[compactID];
        voxel_pdf = PdfTraverseLightTree(0, leafID, hit.position,
                                         hit.geometryNormal, float3(0),
                                         teconfig, tree_nodes);
    }
    else if (config.type == VoxelGuidingType::VG_VisibilityIrradiance) {
        const int compactID = inverse_index[vxID];
        if (compactID == -1) return 0.0;
        // const int clusterID = u_vxAssociate[compactID];
        // // const float top_pdf = PdfSampleTopLevelTree(u_topLevelTree, spixelID, clusterID);
        // const int topLevelOffset = spixelID * 64;
        // const float topImportance = u_topLevelTree[topLevelOffset + 1];
        // const float leafImportance = u_topLevelTree[topLevelOffset + 32 + clusterID];
        // const float top_pdf = (topImportance == 0.f) ? 0.f : leafImportance / topImportance;
        // const int clusterRoot = u_clusterRoots[clusterID];
        // // pdf of sample the voxel
        // const int leafID = u_compact2leaf[compactID];
        // double pdf = PdfTraverseLightTree_Intensity(clusterRoot, leafID, u_TreeNodes);
        // float a = u_TreeNodes[leafID].intensity;
        // if (top_pdf <= 0) {
        // }
    }
    float sph_pdf = PdfSampleSphericalVoxel(
        vxID,
        vxFlatten,
        hit,
        p_min,
        p_max,
        info);
    return voxel_pdf * double(sph_pdf);
}

/** Returns true if ray intersects plane, else false. */
bool intersectRayPlane(float3 planeP, float3 planeN, float3 rayP, float3 rayD, out float3 I) {
    // Assuming vectors are all normalized
    float denom = dot(planeN, rayD);
    if (abs(denom) > 0.001f) {
        float3 p0l0 = planeP - rayP;
        float t = dot(p0l0, planeN) / denom;
        I = rayP + t * rayD;
        return (t >= 0);
    }
    I = float3(0);
    return false;
}

/** Returns true if point P is within the AABB, else false. */
bool isPointWithinAABB(in const float3 P, in const float3 aabbMin, in const float3 aabbMax) {
    return aabbMin.x < P.x && P.x < aabbMax.x &&
           aabbMin.y < P.y && P.y < aabbMax.y &&
           aabbMin.z < P.z && P.z < aabbMax.z;
}

/** Returns the closet point on an AABB to a ray. */
float3 closestPointOnAABBRay(
    in_ref(float3) rayOrigin, 
    in_ref(float3) rayDir, 
    in_ref(float3) aabbMin, 
    in_ref(float3) aabbMax
) {
    const float3 dirs[3] = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1) };
    float dMin = k_numeric_limits_float_max;
    float dMax = -k_numeric_limits_float_max;
    float d;
    float3 minIs; // Will be written, since ray is guaranteed to intersect one of the planes!
    float3 Is;

    for (int i = 0; i < 6; i++) {
        const float3 planeN = (i < 3) ? dirs[i % 3] : -dirs[i % 3];
        const float3 planeO = (i < 3) ? aabbMax : aabbMin;
        if (intersectRayPlane(planeO, planeN, rayOrigin, rayDir, Is)) {
            Is = clamp(Is, aabbMin, aabbMax);
            float3 dirIs = normalize(Is - rayOrigin);
            d = dot(rayDir, dirIs);
            if (d > dMax) {
                dMin = d;
                dMax = d;
                minIs = Is;
            }
        }
    }
    return minIs;
}

float3 FindSpecularDirectionInAABB(
    in_ref(float3) position,
    in_ref(float3) normal,
    in_ref(float3) view, // direction to eye at shading point
    in_ref(AABB) aabb,
) {
    // Heuristic for estimating the glossy brdf term:
    // Find the closest point on AABB with regard to the perfect reflection.
    float3 Lp = reflect(-view, normal);
    float3 L; // The direction to estimate the BSDF term
    const float3 epsilon = 0.001f;
    if (isPointWithinAABB(position, aabb.min - epsilon, aabb.max + epsilon))
        L = Lp; // inside the AABB, use perfect reflection direction
    else {
        // outside the AABB, find the closest point on AABB to the perfect reflection
        const float3 Is = closestPointOnAABBRay(position, Lp, aabb.min, aabb.max);
        L = normalize(Is - position);
    }
    return L;
}

float ApproxBSDFWeight(
    in_ref(float3) position,
    in_ref(float3x3) frame,
    in_ref(float3) w_in,
    in_ref(AABB) aabb,
    float Kd,
    float Ks,
    float eta,
    float roughness,
) {
    const float3 w_out = FindSpecularDirectionInAABB(position, frame[2], w_in, aabb);

    const float3 half_vector = normalize(w_in + w_out);
    const float n_dot_h = dot(frame[2], half_vector);
    const float n_dot_in = dot(frame[2], w_in);
    const float n_dot_out = dot(frame[2], w_out);

    // dielectric layer:
    // F_o is the reflection percentage.
    const float F_o = FresnelDielectric(dot(half_vector, w_out), eta);
    const float D = GTR2_NDF(n_dot_h, roughness);
    const float G = IsotropicGGX_Masking(to_local(frame, w_in), roughness) *
                    IsotropicGGX_Masking(to_local(frame, w_out), roughness);
    const float spec_contrib = Ks * (G * F_o * D) / (4 * n_dot_in * n_dot_out);
    // diffuse layer:
    // In order to reflect from the diffuse layer,
    // the photon needs to bounce through the dielectric layers twice.
    // The transmittance is computed by 1 - fresnel.
    const float F_i = FresnelDielectric(dot(half_vector, w_in), eta);
    // Multiplying with Fresnels leads to an overly dark appearance at the
    // object boundaries. Disney BRDF proposes a fix to this -- we will implement this in problem set 1.
    const float diffuse_contrib = (1.f - F_o) * (1.f - F_i) / k_pi * Kd;
    return diffuse_contrib + spec_contrib;
}

#endif // !_SRENDERER_ADDON_VXGUIDING_ITNERFACE_HEADER_