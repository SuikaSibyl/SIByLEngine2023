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
        uint strategy;
    } pConst;
};

[[vk::binding(0, 2)]]
RWTexture2D<float4> u_Color;

Texture3D<float2> uRadOpaVox6D[6];
SamplerState uTex3DSampler;

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

float3 VS2WS(int3 voxel, float extentMax, float3 center, int mipLevel) {
    float3 cs = ((voxel + float3(0.5)) / (gVD.voxelSize >> mipLevel)) * 2 - 1;
    return cs * extentMax + center;
}

int2 WS2SS(float3 position) {
    float4 cs = mul(float4(position, 1), globalUniform.cameraData.viewProjMat);
    cs.xyz /= cs.w;
    return int2((cs.xy + 1) * 0.5 * pConst.resolution);
}

float3 unidirectional_path_tracing(in_ref(Ray) primaryRay, inout_ref(RandomSamplerState) RNG) {
    RayDesc ray = ToRayDesc(primaryRay);
    
    PrimaryPayload payload;
    payload.RND = RNG;
    // payload.occluded = false;

    uint instanceMask = 0xff;
    uint rayFlags = RAY_FLAG_NONE;
    TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ray, payload);
    RNG = payload.RND;

    float3 radiance = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);

    if (HasHit(payload.hit) == false)
        return radiance;
    else {
        // GeometryInfo geometry = geometries[payload.hit.geometryID];
        // MaterialInfo material = materials[geometry.materialID];
        // float4 albedo = textures[material.baseOrDiffuseTextureIndex].Sample(payload.hit.texcoord, 0);

        // const PolymorphicLightInfo light = lights[0];
        // float3 lightColor = unpackLightColor(light);
        // float3 lightDir = Unorm32OctahedronToUnitVector(light.databyte0);

        // Ray shadowRay = SpawnRay(payload.hit, -lightDir);
        // bool occluded = TraceOccludeRay(shadowRay, RNG, SceneBVH);
        // float visibility = occluded ? 0.0f : 1.0f;
        // lightDir

        // radiance += albedo.xyz * lightColor * saturate(dot(payload.hit.geometryNormal, -lightDir)) * visibility;
        // throughput *= albedo.xyz;
    }

    if (pConst.strategy == 0) {
        float3 dir = normalize(payload.hit.geometryNormal + randomPointInSphere(RNG));
        Ray bounceRay = SpawnRay(payload.hit, dir);
        ray = ToRayDesc(bounceRay);
        payload.RND = RNG;
        TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ray, payload);

        if (HasHit(payload.hit) == false)
            return radiance;
        else {
            GeometryInfo geometry = geometries[payload.hit.geometryID];
            MaterialInfo material = materials[geometry.materialID];
            const float4 albedo = textures[material.baseOrDiffuseTextureIndex].Sample(payload.hit.texcoord);

            const PolymorphicLightInfo light = lights[0];
            float3 lightColor = unpackLightColor(light);
            float3 lightDir = Unorm32OctahedronToUnitVector(light.databyte0);

            Ray shadowRay = SpawnRay(payload.hit, -lightDir);
            bool occluded = TraceOccludeRay(shadowRay, RNG, SceneBVH);
            float visibility = occluded ? 0.0f : 1.0f;
            // // lightDir

            radiance += albedo.xyz * lightColor * saturate(dot(payload.hit.geometryNormal, -lightDir)) * visibility * k_pi;
        }
    }
    else if (pConst.strategy == 1 || pConst.strategy == 2 || pConst.strategy == 3 || pConst.strategy == 4) {
        int mipLevelShown = 5;
        const int3 voxelSize = gVD.voxelSize >> mipLevelShown;
        const float3 extent = gVD.aabbMax - gVD.aabbMin;
        const float extentMax = max(extent.x, max(extent.y, extent.z)) * 0.5 + 0.01;
        const float3 center = (gVD.aabbMax + gVD.aabbMin) * 0.5;
        const float3 rayPos = ((payload.hit.position.xyz - center) / extentMax + 1) * 0.5 * voxelSize; // [-1, 1]^3

        float pdf = 1.f;
        VoxelTexInfo info = CreateVoxelTexInfo(gVD);
        int3 selectedVox;
        if (pConst.strategy == 1) {
            selectedVox = SampleByLuminance(uRadOpaVox6D, payload.hit.position.xyz, mipLevelShown, info, RNG, pdf);
        }
        else if (pConst.strategy == 2) {
            selectedVox = SampleByLuminanceOcclusion(uRadOpaVox6D,
                                                     uTex3DSampler,
                                                     payload.hit.position.xyz,
                                                     payload.hit.geometryNormal, mipLevelShown, info, RNG, pdf);
            // VXGuidingSetting gs;
            // gs.considerDistanceTerm = false;
            // selectedVox = SampleByEstimation(uRadOpaVox6D,
            //                                  uTex3DSampler,
            //                                  payload.hit.position.xyz,
            //                                  payload.hit.geometryNormal,
            //                                  mipLevelShown,
            //                                  info,
            //                                  RNG,
            //                                  pdf,
            //                                  gs);
        }
        else if (pConst.strategy == 3) {
            VXGuidingSetting gs;
            gs.considerDistanceTerm = true;
            selectedVox = SampleByEstimation(uRadOpaVox6D,
                                             uTex3DSampler,
                                             payload.hit.position.xyz,
                                             payload.hit.geometryNormal,
                                             mipLevelShown,
                                             info,
                                             RNG,
                                             pdf,
                                             gs);
        }
        else if (pConst.strategy == 4) {
            VXGuidingSetting gs;
            gs.considerDistanceTerm = true;
            selectedVox = SampleByEstimationTwoPass(uRadOpaVox6D,
                                                    uTex3DSampler,
                                                    payload.hit.position.xyz,
                                                    payload.hit.geometryNormal,
                                                    mipLevelShown,
                                                    info,
                                                    RNG,
                                                    pdf,
                                                    gs);
        }
        if (all(selectedVox == int3(-1))) return float3(1, 0, 1); // invalid voxel
        if (all(selectedVox == int3(-2))) return float3(1, 0, 0); // invalid voxel

        float voxelExtent = extentMax / 64;
        float3 voxelCenter = VS2WS(selectedVox, extentMax, center, 0);
        AABB aabb = { voxelCenter - voxelExtent, voxelCenter + voxelExtent };

        float2 uv = float2(GetNextRandom(RNG), GetNextRandom(RNG));

        float sph_pdf;
        float3 dir = SampleSphericalVoxel(
            voxelCenter,
            voxelExtent,
            payload.hit.position,
            float3(GetNextRandom(RNG), uv),
            sph_pdf);
        
        float3 geometryNormal = normalize(payload.hit.geometryNormal);

        Ray bounceRay = SpawnRay(payload.hit, dir);
        ray = ToRayDesc(bounceRay);
        payload.RND = RNG;
        TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ray, payload);

        if (HasHit(payload.hit) == false)
            return radiance;
        else if (all(payload.hit.position >= aabb.min) && all(payload.hit.position <= aabb.max)) {
            GeometryInfo geometry = geometries[payload.hit.geometryID];
            MaterialInfo material = materials[geometry.materialID];
            const float4 albedo = textures[material.baseOrDiffuseTextureIndex].Sample(payload.hit.texcoord);

            const PolymorphicLightInfo light = lights[0];
            float3 lightColor = unpackLightColor(light);
            float3 lightDir = Unorm32OctahedronToUnitVector(light.databyte0);

            Ray shadowRay = SpawnRay(payload.hit, -lightDir);
            bool occluded = TraceOccludeRay(shadowRay, RNG, SceneBVH);
            float visibility = occluded ? 0.0f : 1.0f;

            radiance += albedo.xyz * lightColor * saturate(dot(payload.hit.geometryNormal, -lightDir)) 
                * saturate(dot(geometryNormal, dir)) * visibility / (pdf * sph_pdf);
        }

        // if (hitSphere(voxelCenter, sphereRadius, primaryRay.origin, primaryRay.direction) > 0) {
        //     radiance = lerp(radiance, float3(1, 0, 0), 0.5);
        // }
    } 

    return radiance;
}

[shader("raygeneration")]
void RgenMain() {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(pConst.resolution);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    // If the pixel is outside of the image, don't do anything:
    if(any(pixel >= resolution)) {
        return;
    }
    RandomSamplerState RNG = InitRandomSampler(pixel, pConst.sample_batch);
    
    Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );

    float3 pixelColor = unidirectional_path_tracing(primaryRay, RNG);
    u_Color[pixel] = float4(pixelColor, 1);
}