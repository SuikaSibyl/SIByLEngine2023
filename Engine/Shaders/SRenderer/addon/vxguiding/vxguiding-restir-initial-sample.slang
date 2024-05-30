#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/primitives/quad.hlsli"
#include "../../raytracer/primitives/sphere.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../restirgi/include/GIReservoir.hlsli "
#include "../restirgi/include/GIResamplingFunctions.hlsli"
#include "../vbuffer/vbuffer.hlsli"
#include "../vxgi/include/conetrace_utils.hlsli"
#include "../vxgi/include/vxgi_interface.hlsli"
#include "include/vxguiding_interface.hlsli"
#include "testdi_interface.hlsli"
#include "tree/shared.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        GIResamplingRuntimeParameters runtimeParams;
        uint2 resolution;
        uint sample_batch;
        uint sampleStragegy;
        uint initialOutputBufferIndex;
    } pConst;
};

RWTexture2D<float4> u_Color;

RWStructuredBuffer<PackedGIReservoir> u_GIReservoirs;
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
    if (any(pixel >= resolution)) return;
    RandomSamplerState RNG = InitRandomSampler(pixel, pConst.sample_batch);
    // Reservoir for ReSTIR
    GIReservoir reservoir = EmptyGIReservoir();
    float selectedTargetPdf = 0.0f;

    // Load Primary Hit from VBuffer
    // --------------------------------------------------------------------
    const Ray primaryRay = generateRay(pixel, resolution, false, globalUniform.cameraData);
    const VBufferData vhit = UnpackVBufferData(u_vBuffer[pixel]);
    if (IsInvalid(vhit)) {
        // If hit nothing, just clear the reservoir and return
        StoreGIReservoir(reservoir, pConst.runtimeParams, pixel, pConst.initialOutputBufferIndex, u_GIReservoirs);
        return;
    }
    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);
    
    // Data preparation for VXPG
    // --------------------------------------------------------------------
    // Prepare payload structure for path tracing
    PrimaryPayload payload;
    payload.RND = RNG;
    // Prepare information for VXPG
    VoxelTexInfo info = CreateVoxelTexInfo(gVD);
    float3 radiance = float3(0, 0, 0);
    const int sample_stragegy = pConst.sampleStragegy & 0xF;
    const int vxg_type = (pConst.sampleStragegy >> 4) & 0x3;
    const bool power_mis = ((pConst.sampleStragegy >> 6) & 0x1) != 0;
    const VisibilityType visibility = VisibilityType((pConst.sampleStragegy >> 7) & 0x3);
    const float2 uv = float2(GetNextRandom(RNG), GetNextRandom(RNG));

    {   // VXPG + ReSTIR GI
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

        if (true) { // use fuzzy clustering - sampling
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
        const int topIndex = SampleTopLevelTree(u_topLevelTree, spixelID, GetNextRandom(RNG), info, top_pdf);

        float max_irradiance = 0.f;
        float4 vpl_pos = float4(0);
        // First draw BSDF samples
        {   
            float bsdf_pdf;
            Ray bsdfRay = SpawnBsdfRay(hit, -primaryRay.direction, RNG, bsdf_pdf);
            const uint32_t RNGState = RNG.state;
            float3 di = float3(0);
            float3 indirect = EvaluateIndirectLightEX(
                primaryRay, bsdfRay,
                double(bsdf_pdf),
                hit, payload, RNG, di);
            // di += EvaluateMultibounceIndirect(bsdfRay, 1, float3(1), payload, RNG);
            
            const float irradiance = maxComponent(di);
            if (irradiance != 0) {
                const uint hitNormal = UnitVectorToUnorm32Octahedron(payload.hit.geometryNormal);
                max_irradiance = irradiance;
                vpl_pos = float4(payload.hit.position, asfloat(hitNormal));
            }
            
            if (true) {
                float w1 = bsdf_pdf;
                float w2 = 0;
                {
                    // pdf of sample top level tree
                    // const int3 vxID = int3(WorldToVoxel(bsdfPos.xyz, info) * info.volumeDimension);
                    const int3 vxID = int3(WorldToVoxel(payload.hit.position.xyz, info) * info.volumeDimension);
                    const int vxFlatten = FlatIndex(vxID, info.volumeDimension);
                    int compactID = u_inverseIndex[vxID];
                    if (compactID == -1) {
                    }
                    else {
                        const int clusterID = u_vxAssociate[compactID];
                        // const float top_pdf = PdfSampleTopLevelTree(u_topLevelTree, spixelID, clusterID);
                        float top_pdf = 0.f;
                        {
                            const int topLevelOffset = spixelID * 64;
                            const float topImportance = u_topLevelTree[topLevelOffset + 1];
                            const float leafImportance = u_topLevelTree[topLevelOffset + 32 + clusterID];
                            top_pdf = (topImportance == 0.f) ? 0.f : leafImportance / topImportance;
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
                float mis_weight = w1 / (w1 + w2);
                radiance += indirect * mis_weight;

                // multibounce
                if (HasHit(payload.hit)) {
                    float3 multi_bounce = float3(0);
                    Ray prev_bsdf_ray = bsdfRay;
                    GeometryHit secondary_hit = payload.hit;
                    // extra bounce testing
                    // MIS the bsdf sampling and voxel guider sampling
                    VoxelGuidingConfig config;
                    config.type = VoxelGuidingType::VG_Irradiance;
                    config.visibility = VisibilityType::None;
                    int VXCount = u_vxCounter.Load(0);
                    const int3 currVXID = int3(WorldToVoxel(secondary_hit.position.xyz, info) * info.volumeDimension);

                    { // BSDF sampling, we really sample the bsdf sampling
                        float bsdf_pdf;
                        Ray bsdf_ray = SpawnBsdfRay(secondary_hit, -prev_bsdf_ray.direction, RNG, bsdf_pdf);
                        float3 gi = EvaluateIndirectLight(
                            prev_bsdf_ray, bsdf_ray,
                            double(bsdf_pdf),
                            secondary_hit, payload, RNG);
                        const float3 indirect = gi;
                        if (any(indirect > 0)) {
                            const int3 vxID = int3(WorldToVoxel(payload.hit.position.xyz, info) * info.volumeDimension);
                            float w1 = bsdf_pdf;
                            float w2 = float(PdfVoxelGuiding(
                                VXCount, vxID, currVXID, secondary_hit, info, config,
                                u_pMin, u_pMax, u_pCompactIndex, u_TreeNodes, u_inverseIndex, u_compact2leaf, u_pIrradiance));
                            if (power_mis) {
                                w1 *= w1;
                                w2 *= w2;
                            }
                            di += indirect * w1 / (w1 + w2);
                        }
                        payload.hit = secondary_hit;
                    }
                    { // Semi NEE - voxel2sphere
                        double pdf;
                        AABB aabb;
                        const float3 guidedDir = SampleVoxelGuiding(
                            VXCount, pixel, uv, currVXID, secondary_hit, info, config, RNG,
                            u_pMin, u_pMax, u_pCompactIndex, u_TreeNodes,
                            u_spixelIdx, u_fuzzyWeight, u_fuzzyIDX, u_topLevelTree, u_clusterRoots,
                            aabb, pdf);
                        Ray guidedRay = SpawnRay(secondary_hit, guidedDir);
                        float3 gi = EvaluateIndirectLight(
                            prev_bsdf_ray, guidedRay,
                            pdf,
                            secondary_hit, payload, RNG);

                        if ((all(payload.hit.position >= aabb.min) && all(payload.hit.position <= aabb.max))) {
                            if (any(gi > 0)) {
                                float w1 = PdfBsdfSample(secondary_hit, -prev_bsdf_ray.direction, guidedRay.direction);
                                float w2 = float(pdf);
                                if (power_mis) {
                                    w1 *= w1;
                                    w2 *= w2;
                                }
                                di += gi * w2 / (w1 + w2);
                            }
                        }
                        payload.hit = secondary_hit;
                    }
                }
                
                {
                    GIReservoir local = MakeGIReservoir(payload.hit.position, payload.hit.geometryNormal, di, bsdf_pdf, RNGState);
                    const float target_pdf = GetGISampleTargetPdfForSurface(local.position, local.radiance, hit, globalUniform.cameraData.posW);
                    if (CombineGIReservoirs(reservoir, local, GetNextRandom(RNG), target_pdf * mis_weight)) {
                        selectedTargetPdf = target_pdf;
                    }
                }
            }
        }
        // Then draw voxel samples
        if (topIndex != -1) {
            const int clusterRoot = u_clusterRoots[topIndex];
            TreeEvaluateConfig config;
            int VXCount = u_vxCounter.Load(0);
            double pdf;
            int selectedID = TraverseLightTree(clusterRoot, VXCount - 1, hit.position, hit.geometryNormal, float3(0), GetNextRandom(RNG), pdf, config, u_TreeNodes);

            if (selectedID == -1) {
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
                float3 di;
                float3 gi = EvaluateIndirectLightEX(
                    primaryRay, guidedRay,
                    pdf * double(sph_pdf),
                    hit, payload, RNG, di);
                di += EvaluateMultibounceIndirect(guidedRay, 1, float3(1), payload, RNG);
                
                // const float irradiance = maxComponent(di);
                // if (irradiance != 0 && max_irradiance == 0) {
                //     const uint hitNormal = UnitVectorToUnorm32Octahedron(payload.hit.geometryNormal);
                //     max_irradiance = irradiance;
                //     vpl_pos = float4(payload.hit.position, asfloat(hitNormal));
                // }
                
                if ((all(payload.hit.position >= aabb.min) && all(payload.hit.position <= aabb.max))) {
                    float w1 = PdfBsdfSample(hit, -primaryRay.direction, guidedDir);
                    float w2 = float(pdf * double(sph_pdf));
                    w1 = w1 * w1;
                    w2 = w2 * w2;
                    if (isnan(w1)) w1 = 0;
                    float mis_weight = w2 / (w1 + w2);
                    radiance += gi * mis_weight;

                    {
                        float guided_pdf = float(pdf * double(sph_pdf));
                        GIReservoir local = MakeGIReservoir(payload.hit.position, payload.hit.geometryNormal, di, guided_pdf, 0);
                        const float target_pdf = GetGISampleTargetPdfForSurface(local.position, local.radiance, hit, globalUniform.cameraData.posW);
                        if (CombineGIReservoirs(reservoir, local, GetNextRandom(RNG), target_pdf * mis_weight)) {
                            selectedTargetPdf = target_pdf;
                        }
                    }
                }
            }
        }

        u_intensity[pixel] = max_irradiance;
        u_position[pixel] = vpl_pos;
    }

    if (any(isnan(radiance))) {
        radiance = float3(1000, 0, 1000);
    }
    u_Color[pixel] = float4(radiance, 1);
    
    // if (reservoir.M != 0)
    reservoir.M = 1;
    const float normalizationNumerator = 1.0;
    const float normalizationDenominator = selectedTargetPdf;
    FinalizeGIResampling(reservoir, normalizationNumerator, normalizationDenominator);
    StoreGIReservoir(reservoir, pConst.runtimeParams, pixel, pConst.initialOutputBufferIndex, u_GIReservoirs);
}