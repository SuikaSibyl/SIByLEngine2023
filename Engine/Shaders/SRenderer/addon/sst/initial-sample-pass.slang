#include "../../raytracer/spt_interface.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"

#include "seminee_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
    } pConst;
};

RWStructuredBuffer<VPLData> u_VPLs;

VPLData initial_sample(in_ref(Ray) primaryRay, inout_ref(RandomSamplerState) RNG) {
    RayDesc ray = ToRayDesc(primaryRay);

    PrimaryPayload payload;
    payload.RND = RNG;

    uint instanceMask = 0xff;
    uint rayFlags = RAY_FLAG_NONE;
    TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ray, payload);
    RNG = payload.RND;

    float3 radiance = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);

    VPLData vpl;
    vpl.id = -1;

    if (HasHit(payload.hit) == false) {
        return vpl;
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

    // cosineSampleHemisphere
    // GeometryInfo geometry = geometries[payload.hit.geometryID];
    // MaterialInfo material = materials[geometry.materialID];

    float3 dir = normalize(payload.hit.geometryNormal + randomPointInSphere(RNG));
    Ray bounceRay = SpawnRay(payload.hit, dir);
    ray = ToRayDesc(bounceRay);
    payload.RND = RNG;
    TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ray, payload);

    if (HasHit(payload.hit) == false)
        return vpl;
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
        vpl.setPosW(payload.hit.position);
        vpl.setAABBMax(payload.hit.position);
        vpl.setAABBMin(payload.hit.position);
        vpl.setNormW(payload.hit.geometryNormal);
        vpl.setVariance(float3(0, 0, 0));
        vpl.setIntensity(luminance(color));
        vpl.setColor(color);
        vpl.setEarlyStop(0.f);
        if (radiance > 0.0f)
            vpl.id = 0;
    }
    return vpl;
}

[shader("raygeneration")]
void InitialSampleRgen() {
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

    VPLData vpl = initial_sample(primaryRay, RNG);
    if (vpl.id != -1)
        vpl.id = pixel.y * resolution.x + pixel.x;
    u_VPLs[pixel.y * resolution.x + pixel.x] = vpl;

    vpl.id = -1;
    u_VPLs[pixel.y * resolution.x + pixel.x + resolution.x * resolution.y] = vpl;
}