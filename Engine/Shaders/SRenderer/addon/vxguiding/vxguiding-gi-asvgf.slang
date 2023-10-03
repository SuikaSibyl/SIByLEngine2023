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
#include "../gbuffer/gbuffer_common.hlsli"
#include "../gbuffer/gbuffer_interface.hlsli"
#include "include/vxguiding_interface.hlsli"
#include "testdi_interface.hlsli"
#include "tree/shared.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
        uint sampleStragegy;
        bool secondbounce;
    } pConst;
};

RWTexture2D<uint> u_Diffuse;
RWTexture2D<uint> u_Specular;

RWTexture2D<uint> u_RNGSeed;
RWTexture2D<uint> u_RNGPrev;

RWTexture2D<float4> u_position;
RWTexture2D<float> u_intensity;

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
    RandomSamplerState RNG;
    RNG.state = u_RNGSeed[pixel];
    uint beginState = RNG.state;
    u_RNGPrev[pixel] = RNG.state;

    const Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );

    ShadingSurface surface = GetGBufferSurface(
        pixel, globalUniform.cameraData
    );

    SplitShading shading;
    shading.diffuse = float3(0);
    shading.specular = float3(0);

    if (IsShadingSurfaceValid(surface)) {
        PrimaryPayload payload;
        payload.RND = RNG;
        VoxelTexInfo info = CreateVoxelTexInfo(gVD);
        float3 radiance = float3(0, 0, 0);

        const int sample_stragegy = pConst.sampleStragegy & 0xF;
        const int vxg_type = (pConst.sampleStragegy >> 4) & 0x3;
        const bool power_mis = ((pConst.sampleStragegy >> 6) & 0x1) != 0;
        const VisibilityType visibility = VisibilityType((pConst.sampleStragegy >> 7) & 0x3);

        // 1 spp : spawn a new BSDF ray.
        if (sample_stragegy == 5 || sample_stragegy == 6 || sample_stragegy == 7) {
            float max_irradiance = 0.f;
            float4 vpl_pos = float4(0);
            float3 bsdf_indirect = float3(0);
            // spawn bsdf ray first, to ensure RNG is consistent
            {
                // // new BSDF ray
                // float bsdf_pdf;
                // Ray bsdfRay = SpawnBsdfRay(surface, -primaryRay.direction, RNG, bsdf_pdf);
                // float3 di = float3(0);
                // SplitShading throughput;
                // float3 indirect = EvaluateIndirectLightEXXSplit(
                //     primaryRay, bsdfRay,
                //     double(bsdf_pdf),
                //     surface, payload, RNG, di, throughput);
                // if (bsdf_pdf == 0) {
                //     throughput.diffuse = float3(0);
                //     throughput.specular = float3(0);
                // }

                // const float irradiance = maxComponent(di);
                // if (irradiance != 0) {
                //     const uint hitNormal = UnitVectorToUnorm32Octahedron(payload.hit.geometryNormal);
                //     max_irradiance = irradiance;
                //     vpl_pos = float4(payload.hit.position, asfloat(hitNormal));
                // }
                
                // shading.diffuse += throughput.diffuse * indirect;
                // shading.specular += throughput.specular * indirect;

                float bsdf_pdf;
                Ray bsdfRay = SpawnBsdfRay(surface, -primaryRay.direction, RNG, bsdf_pdf);
                SplitShading gi = EvalBsdfSplit(surface, -primaryRay.direction, bsdfRay.direction);

                PrimaryPayload payload;
                Intersection(bsdfRay, SceneBVH, payload, RNG);
                if (HasHit(payload.hit)) {
                    const PolymorphicLightInfo light = lights[0];
                    float3 radiance = EvaluateDirectLight(bsdfRay, payload.hit, light, RNG);
                    // further bounces
                    // radiance += EvaluateMultibounceIndirect(bsdfRay, 1, throughput, payload, RNG);

                    shading.diffuse += radiance * gi.diffuse / bsdf_pdf;
                    shading.specular += radiance * gi.specular / bsdf_pdf;
                }
            }
            // {
            //     if (any(indirect > 0)) {
            //         float w1 = bsdf_pdf;
            //         float w2 = 0;
            //         {
            //             // pdf of sample top level tree
            //             // const int3 vxID = int3(WorldToVoxel(bsdfPos.xyz, info) * info.volumeDimension);
            //             const int3 vxID = int3(WorldToVoxel(payload.hit.position.xyz, info) * info.volumeDimension);
            //             const int vxFlatten = FlatIndex(vxID, info.volumeDimension);
            //             int compactID = u_inverseIndex[vxID];
            //             if (compactID == -1) {
            //                 // u_Color[pixel] = float4(0, 1000, 1000, 1);
            //                 // return;
            //             }
            //             else {
            //                 const int clusterID = u_vxAssociate[compactID];
            //                 // const float top_pdf = PdfSampleTopLevelTree(u_topLevelTree, spixelID, clusterID);
            //                 float top_pdf = 0.f;
            //                 if (sample_stragegy == 5 || sample_stragegy == 6) {
            //                     const int topLevelOffset = spixelID * 64;
            //                     const float topImportance = u_topLevelTree[topLevelOffset + 1];
            //                     const float leafImportance = u_topLevelTree[topLevelOffset + 32 + clusterID];
            //                     top_pdf = (topImportance == 0.f) ? 0.f : leafImportance / topImportance;
            //                 }
            //                 else if (sample_stragegy == 7) {
            //                     float accum = 0.f;
            //                     // float4 debug0;
            //                     // float4 debug1;
            //                     // float4 debug2;
            //                     {
            //                         const int topLevelOffset = spixelID * 64;
            //                         const float topImportance = u_topLevelTree[topLevelOffset + 1];
            //                         const float leafImportance = u_topLevelTree[topLevelOffset + 32 + clusterID];
            //                         // const float debug = (topImportance == 0.f) ? 0.f : leafImportance / topImportance;
            //                         // debug2 = float4(spixelID, clusterID, leafImportance, topImportance);
            //                     }
            //                     for (int i = 0; i < 4; ++i) {
            //                         if (weights[i] > 0.f) {
            //                             spixelID = indices[i];
            //                             const int topLevelOffset = spixelID * 64;
            //                             const float topImportance = parent_weights[i];
            //                             const float leafImportance = u_topLevelTree[topLevelOffset + 32 + clusterID];
            //                             top_pdf += (topImportance == 0.f) ? 0.f : leafImportance / topImportance * weights[i];
            //                             // debug1[i] = leafImportance;
            //                             // debug0[i] = topImportance;
            //                         }
            //                         else {
            //                             // debug0[i] = 0;
            //                             // debug1[i] = 0;
            //                         }
            //                     }
            //                     // u_Debug[pixel] = debug0;
            //                     // u_Debug2[pixel] = debug1;
            //                     // u_Debug3[pixel] = debug2;
            //                 }
            //                 const int clusterRoot = u_clusterRoots[clusterID];
            //                 // pdf of sample the voxel
            //                 const int leafID = u_compact2leaf[compactID];
            //                 double pdf = PdfTraverseLightTree_Intensity(clusterRoot, leafID, u_TreeNodes);
            //                 float a = u_TreeNodes[leafID].intensity;
            //                 if (top_pdf <= 0) {
            //                 }
            //                 pdf *= double(top_pdf);

            //                 // pdf of sample the spherical voxel
            //                 float sph_pdf = PdfSampleSphericalVoxel(
            //                     vxID,
            //                     vxFlatten,
            //                     surface,
            //                     u_pMin,
            //                     u_pMax,
            //                     info);
            //                 pdf *= double(sph_pdf);
            //                 w2 = float(pdf);
            //                 if (isnan(w2)) w2 = 0;
            //             }
            //         }
            //         w1 = w1 * w1;
            //         w2 = w2 * w2;
            //         radiance += indirect * w1 / (w1 + w2);
            //         shading.diffuse += throughput.diffuse * indirect;
            //         shading.specular += throughput.specular * indirect;
            //     }
            // }
            
            // // Semi NEE - voxel2sphere
            // int spixelID = u_spixelIdx[pixel];
            // float4 weights = u_fuzzyWeight[pixel];
            // int4 indices = u_fuzzyIDX[pixel];
            // float4 parent_weights;
            // for (int i = 0; i < 4; ++i) {
            //     parent_weights[i] = (weights[i] > 0) ?
            //  u_topLevelTree[indices[i] * 64 + 1] : 0;
            //     if (parent_weights[i] == 0) weights[i] = 0;
            // }
            // weights /= dot(weights, float4(1));

            // if (sample_stragegy == 6 || sample_stragegy == 7) {
            //     float fuzzy_rnd = GetNextRandom(RNG);
            //     float accum = 0.f;
            //     for (int i = 0; i < 4; ++i) {
            //         accum += weights[i];
            //         if (fuzzy_rnd < accum) {
            //             spixelID = indices[i];
            //             break;
            //         }
            //     }
            // }

            // // sample top level tree
            // double top_pdf = 1.f;
            // const int topIndex = SampleTopLevelTree(u_topLevelTree, spixelID, GetNextRandom(RNG), top_pdf);

            // // {

            // //     if (any(indirect > 0)) {
            // //         float w1 = bsdf_pdf;
            // //         float w2 = 0;
            // //         {
            // //             // pdf of sample top level tree
            // //             // const int3 vxID = int3(WorldToVoxel(bsdfPos.xyz, info) * info.volumeDimension);
            // //             const int3 vxID = int3(WorldToVoxel(payload.hit.position.xyz, info) * info.volumeDimension);
            // //             const int vxFlatten = FlatIndex(vxID, info.volumeDimension);
            // //             int compactID = u_inverseIndex[vxID];
            // //             if (compactID == -1) {
            // //                 // u_Color[pixel] = float4(0, 1000, 1000, 1);
            // //                 // return;
            // //             }
            // //             else {
            // //                 const int clusterID = u_vxAssociate[compactID];
            // //                 // const float top_pdf = PdfSampleTopLevelTree(u_topLevelTree, spixelID, clusterID);
            // //                 float top_pdf = 0.f;
            // //                 if (sample_stragegy == 5 || sample_stragegy == 6) {
            // //                     const int topLevelOffset = spixelID * 64;
            // //                     const float topImportance = u_topLevelTree[topLevelOffset + 1];
            // //                     const float leafImportance = u_topLevelTree[topLevelOffset + 32 + clusterID];
            // //                     top_pdf = (topImportance == 0.f) ? 0.f : leafImportance / topImportance;
            // //                 }
            // //                 else if (sample_stragegy == 7) {
            // //                     float accum = 0.f;
            // //                     // float4 debug0;
            // //                     // float4 debug1;
            // //                     // float4 debug2;
            // //                     {
            // //                         const int topLevelOffset = spixelID * 64;
            // //                         const float topImportance = u_topLevelTree[topLevelOffset + 1];
            // //                         const float leafImportance = u_topLevelTree[topLevelOffset + 32 + clusterID];
            // //                         // const float debug = (topImportance == 0.f) ? 0.f : leafImportance / topImportance;
            // //                         // debug2 = float4(spixelID, clusterID, leafImportance, topImportance);
            // //                     }
            // //                     for (int i = 0; i < 4; ++i) {
            // //                         if (weights[i] > 0.f) {
            // //                             spixelID = indices[i];
            // //                             const int topLevelOffset = spixelID * 64;
            // //                             const float topImportance = parent_weights[i];
            // //                             const float leafImportance = u_topLevelTree[topLevelOffset + 32 + clusterID];
            // //                             top_pdf += (topImportance == 0.f) ? 0.f : leafImportance / topImportance * weights[i];
            // //                             // debug1[i] = leafImportance;
            // //                             // debug0[i] = topImportance;
            // //                         }
            // //                         else {
            // //                             // debug0[i] = 0;
            // //                             // debug1[i] = 0;
            // //                         }
            // //                     }
            // //                     // u_Debug[pixel] = debug0;
            // //                     // u_Debug2[pixel] = debug1;
            // //                     // u_Debug3[pixel] = debug2;
            // //                 }
            // //                 const int clusterRoot = u_clusterRoots[clusterID];
            // //                 // pdf of sample the voxel
            // //                 const int leafID = u_compact2leaf[compactID];
            // //                 double pdf = PdfTraverseLightTree_Intensity(clusterRoot, leafID, u_TreeNodes);
            // //                 float a = u_TreeNodes[leafID].intensity;
            // //                 if (top_pdf <= 0) {
            // //                 }
            // //                 pdf *= double(top_pdf);

            // //                 // pdf of sample the spherical voxel
            // //                 float sph_pdf = PdfSampleSphericalVoxel(
            // //                     vxID,
            // //                     vxFlatten,
            // //                     surface,
            // //                     u_pMin,
            // //                     u_pMax,
            // //                     info);
            // //                 pdf *= double(sph_pdf);
            // //                 w2 = float(pdf);
            // //                 if (isnan(w2)) w2 = 0;
            // //             }
            // //         }
            // //         w1 = w1 * w1;
            // //         w2 = w2 * w2;
            // //         radiance += indirect * w1 / (w1 + w2);
            // //         shading.diffuse += throughput.diffuse * indirect;
            // //         shading.specular += throughput.specular * indirect;
            // //     }
            // // }
            // // if (topIndex != -1) {
            // //     const int clusterRoot = u_clusterRoots[topIndex];
            // //     TreeEvaluateConfig config;
            // //     int VXCount = u_vxCounter.Load(0);
            // //     double pdf;
            // //     int selectedID = TraverseLightTree(clusterRoot, VXCount - 1, surface.worldPos, surface.geometryNormal, float3(0), GetNextRandom(RNG), pdf, config, u_TreeNodes);

            // //     if (selectedID == -1) {
            // //         radiance = float3(0);
            // //     }
            // //     else {
            // //         int tmp = selectedID;
            // //         selectedID = u_TreeNodes[selectedID].vx_idx;
            // //         float bottom_pdf = float(pdf);
            // //         pdf *= double(top_pdf);
            // //         int vxFlatten = u_pCompactIndex[selectedID];

            // //         // Semi NEE - voxel2sphere
            // //         AABB aabb;
            // //         float sph_pdf;
            // //         float3 guidedDir = SampleSphericalVoxel(
            // //             vxFlatten,
            // //             float3(GetNextRandom(RNG), uv),
            // //             surface,
            // //             u_pMin,
            // //             u_pMax,
            // //             info,
            // //             aabb,
            // //             sph_pdf);

            // //         Ray guidedRay = SpawnRay(surface, guidedDir);
            // //         float3 di;
            // //         float3 gi = EvaluateIndirectLightEX(
            // //             primaryRay, guidedRay,
            // //             pdf * double(sph_pdf),
            // //             surface, payload, RNG, di);
                        
            // //         // const float irradiance = maxComponent(di);
            // //         // if (irradiance != 0 && max_irradiance == 0) {
            // //         //     const uint hitNormal = UnitVectorToUnorm32Octahedron(payload.hit.geometryNormal);
            // //         //     max_irradiance = irradiance;
            // //         //     vpl_pos = float4(payload.hit.position, asfloat(hitNormal));
            // //         // }

            // //         if ((all(payload.hit.position >= aabb.min) && all(payload.hit.position <= aabb.max))) {
            // //             float w1 = PdfBsdfSample(surface, -primaryRay.direction, guidedDir);
            // //             float w2 = float(pdf * double(sph_pdf));
            // //             w1 = w1 * w1;
            // //             w2 = w2 * w2;
            // //             if (isnan(w1)) w1 = 0;
            // //             radiance += gi * w2 / (w1 + w2);
            // //         }
            // //     }
            // // }

            u_intensity[pixel] = max_irradiance;
            u_position[pixel] = vpl_pos;
        }
    }

    if (IsShadingSurfaceValid(surface)) {
        // Add direct light
        const PolymorphicLightInfo light = lights[0];
        SplitShading di = EvaluateDirectLightSplit(primaryRay, surface, light, RNG);
        shading = add(shading, di);
    }

    RNG = InitRandomSampler(pixel, pConst.sample_batch);
    u_RNGSeed[pixel] = RNG.state;
    u_Diffuse[pixel] = PackRGBE(shading.diffuse);
    u_Specular[pixel] = PackRGBE(shading.specular);
}