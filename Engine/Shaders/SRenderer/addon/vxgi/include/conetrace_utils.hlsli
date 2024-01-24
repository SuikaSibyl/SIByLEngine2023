#ifndef _SRENDERER_ADDON_VXGI_CONETRACE_UTILS_HEADER_
#define _SRENDERER_ADDON_VXGI_CONETRACE_UTILS_HEADER_

#include "../../../include/common/cpp_compatible.hlsli"
#include "../../../include/common/geometry.hlsli"
#include "vxgi_interface.hlsli"

static const int VOXEL_DIMENSION = 64;

#ifndef VOXEL_TYPE
#define VOXEL_TYPE float2
#define ALPHA_CHANNEL 1
#endif

int3 ComputeVoxelFaceIndices(in_ref(float3) direction) {
    return int3(direction.x > 0.0 ? 1 : 0,
                direction.y > 0.0 ? 3 : 2,
                direction.z > 0.0 ? 5 : 4);
}

float2 LoadTex(in_ref(Texture3D<float2>) tex, int3 coord, int mipLevel) {
    return tex.Load(int4(coord, mipLevel));
}

VOXEL_TYPE LoadRadopa(
    in_ref(Texture3D<VOXEL_TYPE>) tex[6],
    int3 coord, int mipLevel, in_ref(float3) direction)
{
    const int3 faceIndices = ComputeVoxelFaceIndices(direction);
    const float3 weight = dot(direction, direction);
    // return LoadTex(tex[faceIndices.x], coord, mipLevel) * weight.x
    //      + LoadTex(tex[faceIndices.y], coord, mipLevel) * weight.y
    //      + LoadTex(tex[faceIndices.z], coord, mipLevel) * weight.z;

    return (LoadTex(tex[0], coord, mipLevel)
          + LoadTex(tex[1], coord, mipLevel)
          + LoadTex(tex[2], coord, mipLevel)
          + LoadTex(tex[3], coord, mipLevel)
          + LoadTex(tex[4], coord, mipLevel)
          + LoadTex(tex[5], coord, mipLevel)) / 6;

    // float2 radopa = LoadTex(tex, coord + int3(0, 0, faceIndices.x * faceStride), mipLevel) * weight.x
    //               + LoadTex(tex, coord + int3(0, 0, faceIndices.y * faceStride), mipLevel) * weight.y
    //               + LoadTex(tex, coord + int3(0, 0, faceIndices.z * faceStride), mipLevel) * weight.z;
    // radopa.y = saturate(radopa.y);
    // return radopa;
}

/**
 * Directional sample from anisotropic voxel representation.
 * @param tex 6 faces of anisotropic voxel representation
 * @param sampler sampler state
 * @param coord 3D texture coordinate in [0, 1]
 * @param weight weight per axis for aniso sampling
 * @param face visible face indices
 * @param lod mip level
 */
VOXEL_TYPE AnistropicSample(
    in_ref(Texture3D<VOXEL_TYPE>) tex[6],
    in_ref(SamplerState) sampler,
    in_ref(float3) coord,
    in_ref(float3) weight,
    in_ref(int3) face,
    float lod
) {
    // sample from 3 visible faces and blend with weights given.
    int mipLevel = int(lod);
    int4 vcoord = int4(int3(coord * (64 >> int(mipLevel))), int(mipLevel));
    float3 rcoord = (vcoord.xyz + float3(0.5)) / (64 >> int(mipLevel));
    // return weight.x * tex[face.x].Load(vcoord)
    //      + weight.y * tex[face.y].Load(vcoord)
    //      + weight.z * tex[face.z].Load(vcoord);
    float2 v0 = tex[face.x].SampleLevel(sampler, coord, int(mipLevel));
    float2 v1 = tex[face.y].SampleLevel(sampler, coord, int(mipLevel));
    float2 v2 = tex[face.z].SampleLevel(sampler, coord, int(mipLevel));
    float2 wv = weight.x * v0
              + weight.y * v1
              + weight.z * v2;
    // float l = tex[0].SampleLevel(sampler, coord, int(mipLevel)).x
    //     + tex[1].SampleLevel(sampler, coord, int(mipLevel)).x
    //     + tex[2].SampleLevel(sampler, coord, int(mipLevel)).x
    //     + tex[3].SampleLevel(sampler, coord, int(mipLevel)).x
    //     + tex[4].SampleLevel(sampler, coord, int(mipLevel)).x
    //     + tex[5].SampleLevel(sampler, coord, int(mipLevel)).x;
    // wv.x = l;
    return wv;

    // return weight.x * tex[face.x].SampleLevel(sampler, coord, lod)
    //      + weight.y * tex[face.y].SampleLevel(sampler, coord, lod)
    //      + weight.z * tex[face.z].SampleLevel(sampler, coord, lod);
}

float AnistropicSampleOcclusion(
    in_ref(Texture3D<VOXEL_TYPE>) tex[6],
    in_ref(SamplerState) sampler,
    in_ref(float3) coord,
    in_ref(float3) weight,
    in_ref(int3) face,
    float lod
) {
    // sample from 3 visible faces and blend with weights given.
    int mipLevel = int(lod);
    float v0 = tex[face.x].SampleLevel(sampler, coord, int(mipLevel)).y;
    float v1 = tex[face.y].SampleLevel(sampler, coord, int(mipLevel)).y;
    float v2 = tex[face.z].SampleLevel(sampler, coord, int(mipLevel)).y;
    return dot(weight, float3(v0, v1, v2));
}

VOXEL_TYPE AnistropicSampleConservative(
    in_ref(Texture3D<VOXEL_TYPE>) tex[6],
    in_ref(SamplerState) sampler,
    in_ref(float3) coord,
    in_ref(float3) weight,
    in_ref(int3) face,
    float lod
) {
    // sample from 3 visible faces and blend with weights given.
    int mipLevel = int(lod);
    int4 vcoord = int4(int3(coord * (64 >> int(mipLevel))), int(mipLevel));
    float3 rcoord = (vcoord.xyz + float3(0.5)) / (64 >> int(mipLevel));
    // return weight.x * tex[face.x].Load(vcoord)
    //      + weight.y * tex[face.y].Load(vcoord)
    //      + weight.z * tex[face.z].Load(vcoord);

    return weight.x * tex[face.x].SampleLevel(sampler, coord, int(mipLevel))
         + weight.y * tex[face.y].SampleLevel(sampler, coord, int(mipLevel))
         + weight.z * tex[face.z].SampleLevel(sampler, coord, int(mipLevel));
    // return weight.x * tex[face.x].SampleLevel(sampler, coord, lod)
    //      + weight.y * tex[face.y].SampleLevel(sampler, coord, lod)
    //      + weight.z * tex[face.z].SampleLevel(sampler, coord, lod);
}

/**
* Voxel texture info.
* @param minPoint min point of voxel grid in world space
* @param voxelScale voxel scale
*/
struct VoxelTexInfo {
    float3 minPoint;
    float voxelScale;
    int volumeDimension;
};

VoxelTexInfo CreateVoxelTexInfo(in_ref(VoxerlizerData) data) {
    const float3 extent = data.aabbMax - data.aabbMin;
    const float extentMax = max(extent.x, max(extent.y, extent.z)) * 0.5 + 0.01;
    const float3 center = (data.aabbMax + data.aabbMin) * 0.5;
    VoxelTexInfo info;
    info.minPoint = center - float3(extentMax);
    info.voxelScale = 1.0f / (extentMax * 2.0f);
    info.volumeDimension = data.voxelSize;
    return info;
}

AABB VoxelToBound(
    in_ref(int3) voxelCoord,
    in_ref(int) mipLevel,
    in_ref(VoxelTexInfo) info
) {
    const int voxel_count = 1 << mipLevel;
    AABB bound;
    bound.min = info.minPoint + (voxelCoord + 0) * voxel_count * 1.0f / (info.voxelScale * info.volumeDimension);
    bound.max = info.minPoint + (voxelCoord + 1) * voxel_count * 1.0f / (info.voxelScale * info.volumeDimension);
    return bound;
} 

/**
* Convert world space position to voxel space.
* @param position world space position
* @param info voxel texture info
*/
float3 WorldToVoxel(in_ref(float3) position, in_ref(VoxelTexInfo) info) {
    const float3 voxelPos = position - info.minPoint;
    return voxelPos * info.voxelScale;
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
* @param maxDistWS max distance in world space
* @param samplingFactor sampling factor
*/
VOXEL_TYPE TraceCone(
    in_ref(Texture3D<VOXEL_TYPE>) tex[6],
    in_ref(SamplerState) sampler,
    in_ref(float3) position,
    in_ref(float3) normal,
    in_ref(float3) direction,
    in_ref(float) tan_half_aperture,
    in_ref(VoxelTexInfo) info,
    in_ref(float) maxDistWS,
    in_ref(float) samplingFactor = 1.0f
) {
    // Compute visible face indices
    const int3 visibleFace = ComputeVoxelFaceIndices(direction);
    // world space grid voxel size
    float voxelWorldSize = 1.0 / (info.voxelScale * info.volumeDimension);
    // weight per axis for aniso sampling
    const float3 weight = direction * direction;
    // move further to avoid self collision
    float dist = voxelWorldSize;    // move 1 voxel size to avoid self collision
    const float3 startPosition = position + normal * dist;
    // final results
    float2 coneSample = float2(0.0f);
    float occlusion = 0.0f;
    float maxDistance = maxDistWS / info.voxelScale;
    // out of boundaries check
    float enter = 0.0;
    float leave = 0.0;
    // ray marching loop
    while (coneSample.y < 1.0f && dist <= maxDistance) {
        const float3 conePosition = startPosition + direction * dist;
        // cone expansion and respective mip level based on diameter
        const float diameter = 2.0f * tan_half_aperture * dist;
        const float mipLevel = log2(diameter / voxelWorldSize);
        // convert position to texture coord
        const float3 coord = WorldToVoxel(conePosition, info);
        // get directional sample from anisotropic representation
        VOXEL_TYPE anisoSample = AnistropicSample(tex, sampler, coord, weight, visibleFace, mipLevel);
        // front to back composition, also accumulate occlusion
        coneSample += (1.0f - coneSample[ALPHA_CHANNEL]) * anisoSample;
        // move further into volume
        dist += diameter * samplingFactor;
    }
    return coneSample;
}

// float2 TraceCone_Vox6D(
//     in_ref(float3) origin, in_ref(float3) direction,
//     in_ref(float) tan_half_aperture,
//     in_ref(VoxerlizerData) VD, in_ref(Texture3D<float2>) roTex)
// {
//     const float3 extent = VD.aabbMax - VD.aabbMin;
//     const float extentMax = max(extent.x, max(extent.y, extent.z)) * 0.5 + 0.01;
//     const float3 center = (VD.aabbMax + VD.aabbMin) * 0.5;
//     origin = ((origin.xyz - center) / extentMax + 1) * 0.5 * VD.voxelSize; // [0, 64]^3

//     const float voxel_size = 1;
//     const int MAX_MIP = 3;

//     float distance = 2.0 * voxel_size;
//     float opacity = 0.0;
//     int i = 0;
//     while (opacity < 1.0) {
//         // calculate the cone diameter and mip-level
//         float diameter = 2.0 * tan_half_aperture * distance;
//         int mip_level = int(log2(diameter / voxel_size));
//         if (mip_level > MAX_MIP) {
//             // break;
//         }
//         // get texture coordinate and sample the texture
//         float3 position = origin + distance * direction;
//         int3 texcoord = int3(position);
//         // float4 sample =
//         if (i < 8) {
//             hitVoxels[i++] = int4(texcoord, mip_level);
//         }
//         else break;
//         float voxel_opacity = LoadOpacity(texcoord >> mip_level, mip_level);
//         opacity += (1.0 - opacity) * voxel_opacity;
//         // perform front to back alpha blending

//         // Iterate distance by the cone diameter
//         distance += diameter;
//     }
//     return opacity;
// }

// float2 CastCone(
//     in_ref(float3) startPos, in_ref(float3) direction, 
//     float tan_half_aperture, float maxDistance, float startLevel)
// {
//     // Initialize accumulated luminance and opacity
//     float2 dst = float2(0.0);
//     // Coefficient used in the computation of the diameter of a cone
//     const float coneCoefficient = 2.0 * tan_half_aperture;

//     float curLevel = startLevel;
//     float voxelSize = u_voxelSizeL0 * exp2(curLevel);

//     // Offset startPos in the direction to avoid self occlusion and reduce voxel aliasing
//     startPos += direction * voxelSize * u_traceStartOffset * 0.5;

//     float s = 0.0;
//     float diameter = max(s * coneCoefficient, u_voxelSizeL0);

//     float stepFactor = max(MIN_STEP_FACTOR, u_stepFactor);
//     float occlusion = 0.0;

//     ivec3 faceIndices = computeVoxelFaceIndices(direction); // Implementation in voxelConeTracing/common.glsl
//     vec3 weight = direction * direction;

//     float curSegmentLength = voxelSize;

//     float minRadius = u_voxelSizeL0 * u_volumeDimension * 0.5;

//     // Ray marching - compute occlusion and radiance in one go
//     while (s < maxDistance && occlusion < 1.0)
//     {
//         vec3 position = startPos + direction * s;

//         float distanceToCenter = length(u_volumeCenterL0 - position);
//         float minLevel = ceil(log2(distanceToCenter / minRadius));

//         curLevel = log2(diameter / u_voxelSizeL0);
//         // The startLevel is the minimum level we start off with, minLevel is the current minLevel
//         // It's important to use the max of both (and curLevel of course) because we don't want to suddenly
//         // sample at a lower level than we started off with and ensure that we don't sample in a level that is too low.
//         curLevel = min(max(max(startLevel, curLevel), minLevel), CLIP_LEVEL_COUNT - 1);

//         // Retrieve radiance by accessing the 3D clipmap (voxel radiance and opacity)
//         vec4 radiance = sampleClipmapLinearly(u_voxelRadiance, position, curLevel, faceIndices, weight);
//         float opacity = radiance.a;

//         voxelSize = u_voxelSizeL0 * exp2(curLevel);

//         // Radiance correction
//         float correctionQuotient = curSegmentLength / voxelSize;
//         radiance.rgb = radiance.rgb * correctionQuotient;

//         // Opacity correction
//         opacity = clamp(1.0 - pow(1.0 - opacity, correctionQuotient), 0.0, 1.0);

//         vec4 src = vec4(radiance.rgb, opacity);

//         // Front-to-back compositing
//         dst += clamp(1.0 - dst.a, 0.0, 1.0) * src;
//         occlusion += (1.0 - occlusion) * opacity / (1.0 + (s + voxelSize) * u_occlusionDecay);

//         float sLast = s;
//         s += max(diameter, u_voxelSizeL0) * stepFactor;
//         curSegmentLength = (s - sLast);
//         diameter = s * coneCoefficient;
//     }

//     return clamp(vec4(dst.rgb, 1.0 - occlusion), 0.0, 1.0);
// }

#endif // !_SRENDERER_ADDON_VXGI_CONETRACE_UTILS_HEADER_