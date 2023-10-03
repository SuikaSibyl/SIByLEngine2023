#include "../../../include/common/octahedral.hlsli"
#include "../../../include/common/random.hlsli"
#include "../../../include/raytracer_descriptor_set.hlsli"
#include "../../../include/scene_descriptor_set.hlsli"
#include "../../../raytracer/spt_interface.hlsli"
#include "../../vbuffer/vbuffer.hlsli"
#include "../../vxgi/include/conetrace_utils.hlsli"
#include "../../vxgi/include/vxgi_interface.hlsli"

Texture2D<int2> u_spixel_gathered;
Texture2D<uint> u_spixel_counter;
StructuredBuffer<uint> u_cluster_counter;
StructuredBuffer<float4> u_cluster_gathered;

Texture2D<uint4> u_vBuffer;
RWTexture2D<uint> u_spixel_visibility;
RWTexture2D<float4> u_Debug;

RWStructuredBuffer<uint8_t> u_spixel_avg_visibility;

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    int2 map_size;
    int seed;
};

groupshared uint sharedVisibility[32];
groupshared uint groupVisibility;

bool HandleVisibilityCheck(
    int2 spixelID,
    int2 dispatchID,
    int2 taskOffset,
    int clusterToCheck
) {
    RandomSamplerState RNG = InitRandomSampler(dispatchID, seed);

    // Grab from vsibility buffer
    const uint spixel_count = clamp(u_spixel_counter[spixelID], 0, 1024);
    if (spixel_count == 0) {
        return false;
    }
    const int subtaskFlatten = clamp(int(spixel_count * GetNextRandom(RNG)), 0, spixel_count - 1);
    const int2 subtaskID = int2(subtaskFlatten % 32, subtaskFlatten / 32);
    const int2 pixelID = u_spixel_gathered[taskOffset + subtaskID];

    const Ray primaryRay = generateRay(
        pixelID,
        resolution,
        false,
        globalUniform.cameraData
    );

    const VBufferData vhit = UnpackVBufferData(u_vBuffer[pixelID]);
    if (IsInvalid(vhit)) {
        return false;
    }
    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);

    // Grab a valid vpl from the voxel buffer
    const int clusterVPLCount = clamp(u_cluster_counter[clusterToCheck], 0, 1024);
    if (clusterVPLCount == 0) return false;
    // if ((u_spixel_visibility[spixelID] & uint(1 << clusterToCheck)) != 0) return false;
    const int subtaskFlatten_vpl = clamp(int(clusterVPLCount * GetNextRandom(RNG)), 0, clusterVPLCount - 1);
    const float4 vpl_posnorm = u_cluster_gathered[clusterToCheck * 1024 + subtaskFlatten_vpl];
    const float3 vpl_normal = Unorm32OctahedronToUnitVector(asuint(vpl_posnorm.w));
    float3 dir = vpl_posnorm.xyz - hit.position;
    const float distannce = length(dir);
    dir /= distannce;
    if (dot(-dir, vpl_normal) <= 0.0001 || dot(dir, hit.geometryNormal) <= 0.0001) return false;

    // Visibility test
    bool visible = true;
    Ray visiblityRay = SpawnRay(hit, dir);
    visiblityRay.tMax = distannce - 0.01;

    // Ray query to check visibility
    { RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
        q.TraceRayInline(
            SceneBVH,
            0, 0xff,
            ToRayDesc(visiblityRay));
        if (q.Proceed()) {
            visible = false;
        }
    }
    return visible;
}

[shader("compute")]
[numthreads(32, 8, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int gidx: SV_GroupIndex,
) {
    // each block is handling the sample superpixel
    const int2 dispatchID = int2(DTid.xy);
    const int2 spixelID = int2(dispatchID / 32);
    const int2 task = int2(dispatchID % 32);
    const int2 taskOffset = spixelID * 32;
    const int clusterToCheck = task.y;

    if (all(dispatchID < int2(1280, 720)))
        u_Debug[dispatchID] = float4(WaveGetLaneIndex(), 0, 0, 1);

    const int clusterMod = clusterToCheck % 8;
    if (gidx < 32) {
        groupVisibility = 0;
        sharedVisibility[gidx] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    bool visible = HandleVisibilityCheck(spixelID, dispatchID, taskOffset, clusterToCheck);
    sharedVisibility[clusterToCheck] = WaveActiveSum(visible ? 1 : 0);

    GroupMemoryBarrierWithGroupSync();
    if (gidx < 32) {
        uint32_t visibility = sharedVisibility[gidx];
        if (visibility != 0) {
            InterlockedOr(groupVisibility, uint(1 << gidx));
            const uint spixelIDFlatten = spixelID.y * map_size.x + spixelID.x;
            u_spixel_avg_visibility[spixelIDFlatten * 32 + gidx] = uint8_t(visibility);
        }
    }
    if (gidx == 0) {
        InterlockedOr(u_spixel_visibility[spixelID], groupVisibility);
    }
}