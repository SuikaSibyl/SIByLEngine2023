#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../vxgi/include/conetrace_utils.hlsli"
#include "include/vxguiding_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
        float scaling;
        int2 debug_pixel;
        int use_occlusion;
    } pConst;
};

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

Texture3D<float2> uRadOpaVox6D[6];
SamplerState uTex3DSampler;

RWTexture3D<float2> u_ImportanceMap[6];

float GetImportance(int4 sample, float3 refPoint, float3 refNormal, in_ref(VoxelTexInfo) info) {
    float pdf = 1.;
    for (int i = 5; i >= sample.w; i--) {

        // if (u_ImportanceMap[i][sample.xyz].x != 0) {
        //     return u_ImportanceMap[i][sample.xyz].x;
        // }
    }
    AABB voxel_bound = VoxelToBound(sample.xyz, sample.w, info);
    const float3 direction = normalize((voxel_bound.max + voxel_bound.min) * 0.5 - refPoint);
    float2 radopa = LoadRadopa(uRadOpaVox6D, sample.xyz, sample.w, direction);
    if (pConst.use_occlusion == 1 && sample.w == 5) {
        radopa.x *= TraceConeOcclusionToVoxel(uRadOpaVox6D, uTex3DSampler, refPoint, refNormal, info, voxel_bound, sample.w, 50, 1.0f);
    }
    return radopa.x;
}

void ImportanceInjection(int4 sample, float3 refPoint, float3 refNormal, in_ref(VoxelTexInfo) info) {
    const float importance = GetImportance(sample, refPoint, refNormal, info);
    const float2 value = (importance == 0) ? float2(0, 0) : float2(importance * pConst.scaling, 1);
    u_ImportanceMap[sample.w][sample.xyz] = value;
}

[shader("raygeneration")]
void RgenMain() {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(pConst.resolution);
    RandomSamplerState RNG = InitRandomSampler(
        int2(DispatchRaysIndex().x, DispatchRaysIndex().y * 64 + DispatchRaysIndex().z),
        pConst.sample_batch);

    // Get the reference point
    const int2 pixel = pConst.debug_pixel;
    Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );
    PrimaryPayload payload;
    payload.RND = RNG;
    uint instanceMask = 0xff;
    uint rayFlags = RAY_FLAG_NONE;
    TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ToRayDesc(primaryRay), payload);
    RNG = payload.RND;
    if (HasHit(payload.hit) == false) {
    }
    VoxelTexInfo info = CreateVoxelTexInfo(gVD);

    float3 refPoint = payload.hit.position.xyz;
    float3 refNormal = payload.hit.geometryNormal;

    int3 voxel = DispatchRaysIndex().xyz;
    if (all(voxel.xyz < 2)) {
        ImportanceInjection(int4(voxel.xyz, 5), refPoint, refNormal, info);
    }
    // if (all(voxel.xyz < 4)) {
    //     ImportanceInjection(int4(voxel.xyz, 4), refPoint, refNormal, info);
    // }
    // if (all(voxel.xyz < 8)) {
    //     ImportanceInjection(int4(voxel.xyz, 3), refPoint, refNormal, info);
    // }
    // if (all(voxel.xyz < 16)) {
    //     ImportanceInjection(int4(voxel.xyz, 2), refPoint, refNormal, info);
    // }
    // if (all(voxel.xyz < 32)) {
    //     ImportanceInjection(int4(voxel.xyz, 1), refPoint, refNormal, info);
    // }
    // if (all(voxel.xyz < 64)) {
    //     ImportanceInjection(int4(voxel.xyz, 0), refPoint, refNormal, info);
    // }
}