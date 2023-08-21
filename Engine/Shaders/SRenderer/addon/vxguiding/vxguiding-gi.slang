#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/primitives/quad.hlsli"
#include "../../raytracer/primitives/sphere.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../vbuffer/vbuffer.hlsli"
#include "../vxgi/include/conetrace_utils.hlsli"
#include "../vxgi/include/vxgi_interface.hlsli"
#include "include/vxguiding_interface.hlsli"
#include "testdi_interface.hlsli"
#include "tree/shared.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
        uint sampleStragegy;
        int2 map_size;
    } pConst;
};

RWTexture2D<float4> u_Color;
RWTexture2D<float4> u_Debug;
RWTexture2D<float4> u_Debug2;
RWTexture2D<float4> u_Debug3;
RWTexture2D<float4> u_position;

const Texture3D<int> u_inverseIndex;
const StructuredBuffer<int> u_vxAssociate;
const StructuredBuffer<int> u_compact2leaf;

ByteAddressBuffer u_vxCounter;
const StructuredBuffer<uint> u_pCompactIndex;
const StructuredBuffer<uint4> u_pMin;
const StructuredBuffer<uint4> u_pMax;
const Texture3D<uint> u_pIrradiance;
StructuredBuffer<TreeNode> u_TreeNodes;

const StructuredBuffer<int> u_clusterRoots;
const StructuredBuffer<float> u_topLevelTree;
const Texture2D<int> u_spixelIdx;

const Texture2D<float4> u_fuzzyWeight;
const Texture2D<int4> u_fuzzyIDX;

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

Texture2D<uint4> u_vBuffer;

[shader("raygeneration")]
void RgenMain() {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(pConst.resolution);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) {
        return;
    }
    RandomSamplerState RNG = InitRandomSampler(pixel, pConst.sample_batch);
    const Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );

    const VBufferData vhit = UnpackVBufferData(u_vBuffer[pixel]);
    if (IsInvalid(vhit)) {
        u_Color[pixel] = float4(0, 0, 0, 1);
        return;
    }

    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);
    GeometryInfo geometry = geometries[hit.geometryID];
    MaterialInfo material = materials[geometry.materialID];
    float4 first_albedo = textures[material.baseOrDiffuseTextureIndex].Sample(hit.texcoord, 0);

    PrimaryPayload payload;
    payload.RND = RNG;

    VoxelTexInfo info = CreateVoxelTexInfo(gVD);
    float3 radiance = float3(0, 0, 0);

    const int sample_stragegy = pConst.sampleStragegy & 0xF;
    const int vxg_type = (pConst.sampleStragegy >> 4) & 0x3;
    const bool power_mis = ((pConst.sampleStragegy >> 6) & 0x1) != 0;
    const VisibilityType visibility = VisibilityType((pConst.sampleStragegy >> 7) & 0x3);

    // 1 spp : spawn a new BSDF ray.
    float2 uv = float2(GetNextRandom(RNG), GetNextRandom(RNG));
    if (sample_stragegy == 0) {
        // BSDF
        float bsdf_pdf;
        Ray bsdfRay = SpawnBsdfRay(hit, -primaryRay.direction, RNG, bsdf_pdf);
        float3 gi = EvaluateIndirectLight(
            primaryRay, bsdfRay,
            double(bsdf_pdf),
            hit, payload, RNG);
        radiance += gi;
    }
    // 1 spp - bsdf sampling from vpl
    else if (sample_stragegy == 1) {
        const float4 bsdfPos = u_position[pixel];
        radiance += EvaluateVPLIndirectLight(
            pixel, primaryRay, hit, bsdfPos, u_Color);
    }
    // 2 spp - bsdf sampling (from vpl) + bsdf sampling
    else if (sample_stragegy == 2) {
        {   // the first bsdf sampling comes from the pre sampled vpl
            const float4 bsdfPos = u_position[pixel];
            radiance += EvaluateVPLIndirectLight(
                pixel, primaryRay, hit, bsdfPos, u_Color);
        }
        {   // then we really sample the bsdf sampling
            float bsdf_pdf;
            Ray bsdfRay = SpawnBsdfRay(hit, -primaryRay.direction, RNG, bsdf_pdf);
            float3 gi = EvaluateIndirectLight(
                primaryRay, bsdfRay,
                double(bsdf_pdf),
                hit, payload, RNG);
            radiance += gi;
        }
        if (any(isnan(radiance))) radiance = float3(0);
        radiance /= 2; // 2spp
    }
    else if (sample_stragegy == 3) {
        // voxel guider sampling
        const int3 currVXID = int3(WorldToVoxel(hit.position.xyz, info) * info.volumeDimension);
        int VXCount = u_vxCounter.Load(0);
        // Semi NEE - voxel2sphere
        VoxelGuidingConfig config;
        config.type = VoxelGuidingType(vxg_type);
        config.visibility = visibility;
        double pdf;
        AABB aabb;
        const float3 guidedDir = SampleVoxelGuiding(
            VXCount, pixel, uv, currVXID, hit, info, config, RNG,
            u_pMin, u_pMax, u_pCompactIndex, u_TreeNodes,
            u_spixelIdx, u_fuzzyWeight, u_fuzzyIDX, u_topLevelTree, u_clusterRoots,
            aabb, pdf);
        Ray guidedRay = SpawnRay(hit, guidedDir);
        float3 gi = EvaluateIndirectLight(
            primaryRay, guidedRay,
            pdf,
            hit, payload, RNG);
        
        if ((all(payload.hit.position >= aabb.min) && all(payload.hit.position <= aabb.max))) {
            if (any(gi > 0)) {
                radiance += gi;
            }
        }
    }
    else if (sample_stragegy == 4) {
        // MIS the bsdf sampling and voxel guider sampling
        VoxelGuidingConfig config;
        config.type = VoxelGuidingType(vxg_type);
        config.visibility = visibility;
        int VXCount = u_vxCounter.Load(0);
        const int3 currVXID = int3(WorldToVoxel(hit.position.xyz, info) * info.volumeDimension);
        {   // BSDF sampling, we just use boostrap samples
            const float4 bsdfPos = u_position[pixel];
            float bsdf_pdf;
            const float3 indirect = EvaluateVPLIndirectLight(
                pixel, primaryRay, hit, bsdfPos, u_Color, bsdf_pdf);            
            if (any(indirect > 0)) {
                const int3 vxID = int3(WorldToVoxel(bsdfPos.xyz, info) * info.volumeDimension);
                float w1 = bsdf_pdf;
                float w2 = float(PdfVoxelGuiding(
                    VXCount, vxID, currVXID, hit, info, config,
                    u_pMin, u_pMax, u_pCompactIndex, u_TreeNodes, u_inverseIndex, u_compact2leaf, u_pIrradiance));
                if (power_mis) {
                    w1 *= w1;
                    w2 *= w2;
                }
                radiance += indirect * w1 / (w1 + w2);
            }
        }
        { // Voxel guider sampling
          // voxel guider sampling
            const int3 currVXID = int3(WorldToVoxel(hit.position.xyz, info) * info.volumeDimension);
            // Semi NEE - voxel2sphere
            double pdf;
            AABB aabb;
            const float3 guidedDir = SampleVoxelGuiding(
                VXCount, pixel, uv, currVXID, hit, info, config, RNG,
                u_pMin, u_pMax, u_pCompactIndex, u_TreeNodes,
                u_spixelIdx, u_fuzzyWeight, u_fuzzyIDX, u_topLevelTree, u_clusterRoots,
                aabb, pdf);
            Ray guidedRay = SpawnRay(hit, guidedDir);
            float3 gi = EvaluateIndirectLight(
                primaryRay, guidedRay,
                pdf,
                hit, payload, RNG);
            
            if ((all(payload.hit.position >= aabb.min) && all(payload.hit.position <= aabb.max))) {
                if (any(gi > 0)) {
                    float w1 = PdfBsdfSample(hit, -primaryRay.direction, guidedRay.direction);
                    float w2 = float(pdf);
                    if (power_mis) {
                        w1 *= w1;
                        w2 *= w2;
                    }
                    radiance += gi * w2 / (w1 + w2);
                }
            }
        }
    }
    else if (sample_stragegy == 5 || sample_stragegy == 6 || sample_stragegy == 7) {
        // Semi NEE - voxel2sphere
        int spixelID = u_spixelIdx[pixel];
        float4 weights = u_fuzzyWeight[pixel];
        int4 indices = u_fuzzyIDX[pixel];
        float4 parent_weights;
        for (int i = 0; i < 4; ++i) {
            parent_weights[i] = (weights[i] > 0) ?
             u_topLevelTree[indices[i] * 64 + 1] : 0;
            if (parent_weights[i] == 0) weights[i] = 0;
        }
        weights /= dot(weights, float4(1));

        if (sample_stragegy == 6 || sample_stragegy == 7) {
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
        const int topIndex = SampleTopLevelTree(u_topLevelTree, spixelID, GetNextRandom(RNG), top_pdf);

        if (topIndex != -1) {            
            const int clusterRoot = u_clusterRoots[topIndex];
            TreeEvaluateConfig config;
            int VXCount = u_vxCounter.Load(0);
            double pdf;
            int selectedID = TraverseLightTree(clusterRoot, VXCount - 1, hit.position, hit.geometryNormal, float3(0), GetNextRandom(RNG), pdf, config, u_TreeNodes);

            if (selectedID == -1) {
                radiance = float3(0);
            }
            else {
                int tmp = selectedID;
                selectedID = u_TreeNodes[selectedID].vx_idx;
                float bottom_pdf = float(pdf);
                pdf *= double(top_pdf);
                int vxFlatten = u_pCompactIndex[selectedID];

                // Semi NEE - voxel2sphere
                AABB aabb;
                float sph_pdf;
                float3 guidedDir = SampleSphericalVoxel(
                    vxFlatten,
                    float3(GetNextRandom(RNG), uv),
                    hit,
                    u_pMin,
                    u_pMax,
                    info,
                    aabb,
                    sph_pdf);
                
                Ray guidedRay = SpawnRay(hit, guidedDir);
                float3 gi = EvaluateIndirectLight(
                    primaryRay, guidedRay,
                    pdf * double(sph_pdf),
                    hit, payload, RNG);

                if ((all(payload.hit.position >= aabb.min) && all(payload.hit.position <= aabb.max))) {
                    float w1 = PdfBsdfSample(hit, -primaryRay.direction, guidedDir);
                    float w2 = float(pdf * double(sph_pdf));
                    w1 = w1 * w1;
                    w2 = w2 * w2;
                    radiance += gi * w2 / (w1 + w2);
                }
            }
        }
        {
            const float4 bsdfPos = u_position[pixel];
            float bsdf_pdf;
            const float3 indirect = EvaluateVPLIndirectLight(
                pixel, primaryRay, hit, bsdfPos, u_Color, bsdf_pdf);
            if (any(indirect > 0)) {
                float w1 = bsdf_pdf;
                float w2 = 0;
                {
                    // pdf of sample top level tree
                    const int3 vxID = int3(WorldToVoxel(bsdfPos.xyz, info) * info.volumeDimension);
                    const int vxFlatten = FlatIndex(vxID, info.volumeDimension);
                    int compactID = u_inverseIndex[vxID];
                    if (compactID == -1) {
                        u_Color[pixel] = float4(0, 1000, 1000, 1);
                        return;
                    }
                    else {
                        const int clusterID = u_vxAssociate[compactID];
                        // const float top_pdf = PdfSampleTopLevelTree(u_topLevelTree, spixelID, clusterID);
                        float top_pdf = 0.f;
                        if (sample_stragegy == 5 || sample_stragegy == 6) {
                            const int topLevelOffset = spixelID * 64;
                            const float topImportance = u_topLevelTree[topLevelOffset + 1];
                            const float leafImportance = u_topLevelTree[topLevelOffset + 32 + clusterID];
                            top_pdf = (topImportance == 0.f) ? 0.f : leafImportance / topImportance;
                        }
                        else if (sample_stragegy == 7) {
                            float accum = 0.f;
                            float4 debug0;
                            float4 debug1;
                            float4 debug2;
                            {
                                const int topLevelOffset = spixelID * 64;
                                const float topImportance = u_topLevelTree[topLevelOffset + 1];
                                const float leafImportance = u_topLevelTree[topLevelOffset + 32 + clusterID];
                                const float debug = (topImportance == 0.f) ? 0.f : leafImportance / topImportance;
                                debug2 = float4(spixelID, clusterID, leafImportance, topImportance);
                            }
                            for (int i = 0; i < 4; ++i) {
                                if (weights[i] > 0.f) {
                                    spixelID = indices[i];
                                    const int topLevelOffset = spixelID * 64;
                                    const float topImportance = parent_weights[i];
                                    const float leafImportance = u_topLevelTree[topLevelOffset + 32 + clusterID];
                                    top_pdf += (topImportance == 0.f) ? 0.f : leafImportance / topImportance * weights[i];
                                    debug1[i] = leafImportance;
                                    debug0[i] = topImportance;
                                }
                                else {
                                    debug0[i] = 0;
                                    debug1[i] = 0;
                                }
                            }
                            u_Debug[pixel] = debug0;
                            u_Debug2[pixel] = debug1;
                            u_Debug3[pixel] = debug2;
                        }
                        const int clusterRoot = u_clusterRoots[clusterID];
                        // pdf of sample the voxel
                        const int leafID = u_compact2leaf[compactID];
                        double pdf = PdfTraverseLightTree_Intensity(clusterRoot, leafID, u_TreeNodes);
                        float a = u_TreeNodes[leafID].intensity;
                        if (top_pdf <= 0) {
                        }
                        pdf *= double(top_pdf);

                        // pdf of sample the spherical voxel
                        float sph_pdf = PdfSampleSphericalVoxel(
                            vxID,
                            vxFlatten,
                            hit,
                            u_pMin,
                            u_pMax,
                            info);
                        pdf *= double(sph_pdf);
                        w2 = float(pdf);
                        if (isnan(w2)) w2 = 0;
                    }
                }
                w1 = w1 * w1;
                w2 = w2 * w2;
                radiance += indirect * w1 / (w1 + w2);
            }
        }
    }
    u_Color[pixel] = float4(radiance, 1);
}