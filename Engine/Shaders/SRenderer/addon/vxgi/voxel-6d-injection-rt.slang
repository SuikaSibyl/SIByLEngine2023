#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "include/conetrace_utils.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
    } pConst;
};

RWTexture3D<float2> uRadOpaVox6D[6];

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

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

    float3 radiance = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);

    if (HasHit(payload.hit) == false) {
        return;
    }
    else {
        GeometryInfo geometry = geometries[payload.hit.geometryID];
        MaterialInfo material = materials[geometry.materialID];
        float4 albedo = textures[material.baseOrDiffuseTextureIndex].Sample(payload.hit.texcoord, 0);

        const PolymorphicLightInfo light = lights[0];
        float3 lightColor = unpackLightColor(light);
        float3 lightDir = Unorm32OctahedronToUnitVector(light.databyte0);

        Ray shadowRay = SpawnRay(payload.hit, -lightDir);
        bool occluded = TraceOccludeRay(shadowRay, RNG, SceneBVH);
        float visibility = occluded ? 0.0f : 1.0f;
        // lightDir

        radiance += albedo.xyz * lightColor * saturate(dot(payload.hit.geometryNormal, -lightDir)) * visibility;
        throughput *= albedo.xyz;
    }

    float3 dir = normalize(payload.hit.geometryNormal + randomPointInSphere(RNG));
    Ray bounceRay = SpawnRay(payload.hit, dir);
    payload.RND = RNG;
    TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ToRayDesc(bounceRay), payload);

    if (HasHit(payload.hit) == false)
        return;
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

        const float3 color = albedo.xyz * lightColor * saturate(dot(payload.hit.geometryNormal, -lightDir)) * visibility;
        const float radiance = luminance(color);

        const float3 extent = gVD.aabbMax - gVD.aabbMin;
        const float extentMax = max(extent.x, max(extent.y, extent.z)) * 0.5 + 0.01;
        const float3 center = (gVD.aabbMax + gVD.aabbMin) * 0.5;

        const float3 normalWS = payload.hit.geometryNormal;
        const float3 positionWS = payload.hit.position;
        const float3 positionCS = (positionWS.xyz - center) / extentMax;
        const float3 positionUS = (positionCS + 1.0) * 0.5;
        const int3 voxelID = int3(positionUS * gVD.voxelSize);

        const float3 faceNormals[6] = {
            float3(+1.0, +0.0, +0.0),
            float3(-1.0, +0.0, +0.0),
            float3(+0.0, +1.0, +0.0),
            float3(+0.0, -1.0, +0.0),
            float3(+0.0, +0.0, +1.0),
            float3(+0.0, +0.0, -1.0)
        };

        for (int i = 0; i < 6; ++i) {
            const float projLumin = saturate(dot(normalWS, faceNormals[i]));
            // InterlockedAddF32(uRadOpaVox6D[voxelID + int3(0, 0, 64) * i].x, projLumin);
            uRadOpaVox6D[i][voxelID] = float2(projLumin * luminance(color), 1.0);
        }
        
        // vpl.setPosW(payload.hit.position);
        // vpl.setAABBMax(payload.hit.position);
        // vpl.setAABBMin(payload.hit.position);
        // vpl.setNormW(payload.hit.geometryNormal);
        // vpl.setVariance(float3(0, 0, 0));
        // vpl.setIntensity(luminance(color));
        // vpl.setColor(color);
        // vpl.setEarlyStop(0.f);
        // vpl.numVPLSubTree = 1;
        // if (radiance > 0.0f)
        //     vpl.id = 0;
    }
}