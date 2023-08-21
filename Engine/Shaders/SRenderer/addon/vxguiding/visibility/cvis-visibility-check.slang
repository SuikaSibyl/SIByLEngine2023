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

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    int seed;
};

[shader("raygeneration")]
void RgenMain() {
    const int2 dispatchID = int2(DispatchRaysIndex().xy);
    const int2 spixelID = int2(dispatchID / 32);
    const int2 task = int2(dispatchID % 32);
    const int2 taskOffset = spixelID * 32;
    const int clusterToCheck = task.y;

    RandomSamplerState RNG = InitRandomSampler(dispatchID, seed);

    // Grab from vsibility buffer
    const uint spixel_count = clamp(u_spixel_counter[spixelID], 0, 1024);
    if (spixel_count == 0) {
        return;
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
        return;
    }
    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);

    // Grab a valid vpl from the voxel buffer
    const int clusterVPLCount = clamp(u_cluster_counter[clusterToCheck], 0, 1024);
    if (clusterVPLCount == 0) return;
    if ((u_spixel_visibility[spixelID] & uint(1 << clusterToCheck)) != 0) return;
    const int subtaskFlatten_vpl = clamp(int(clusterVPLCount * GetNextRandom(RNG)), 0, clusterVPLCount - 1);
    const float4 vpl_posnorm = u_cluster_gathered[clusterToCheck * 1024 + subtaskFlatten_vpl];
    const float3 vpl_normal = Unorm32OctahedronToUnitVector(asuint(vpl_posnorm.w));
    float3 dir = vpl_posnorm.xyz - hit.position;
    const float distannce = length(dir);
    dir /= distannce;
    if (dot(-dir, vpl_normal) <= 0.0001 || dot(dir, hit.geometryNormal) <= 0.0001) return;
    
    // Visibility test
    bool visible = true;
    Ray visiblityRay = SpawnRay(hit, dir);
    visiblityRay.tMax = distannce - 0.01;

    u_Debug[pixelID] = float4(visiblityRay.direction, 1);

    if (TraceOccludeRay(visiblityRay, RNG, SceneBVH)) {
        visible = false;
    }

    // Visibility injection
    if (visible)
        InterlockedOr(u_spixel_visibility[spixelID], uint(1 << clusterToCheck));
}