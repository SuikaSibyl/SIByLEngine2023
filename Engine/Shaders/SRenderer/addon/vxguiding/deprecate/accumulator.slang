#include "../../include/common/cylindrical.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/primitives/quad.hlsli"
#include "../../raytracer/primitives/sphere.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../pathguiding/pdf-accumulation/pdf-accumulator.hlsli"
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
        int2 debug_pixel;
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
RWTexture2D<float> u_PdfNormalizing;

[shader("raygeneration")]
void RgenMain() {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(pConst.resolution);
    const int2 pixel = pConst.debug_pixel;
    // If the pixel is outside of the image, don't do anything:
    RandomSamplerState RNG = InitRandomSampler(int2(DispatchRaysIndex().xy), pConst.sample_batch);
    const Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );
    
    const VBufferData vhit = UnpackVBufferData(u_vBuffer[pixel]);
    if (IsInvalid(vhit)) {
        return;
    }

    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);
    GeometryInfo geometry = geometries[hit.geometryID];
    MaterialInfo material = materials[geometry.materialID];

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

            AddOneSample2Info();
            
            Ray guidedRay = SpawnRay(hit, guidedDir);
            float3 gi = EvaluateIndirectLight(
                primaryRay, guidedRay,
                pdf * double(sph_pdf),
                hit, payload, RNG);

            float2 sample = CylindricalToUnit(UnitVectorToCylindrical(guidedDir));
            AddOneSample2Map(int2(sample * 512), 512);
                        
            // if ((all(payload.hit.position >= aabb.min) && all(payload.hit.position <= aabb.max))) {
            //     AddOneSample2Map(int2(sample * 512), 512);
            // }
        }
    }
}