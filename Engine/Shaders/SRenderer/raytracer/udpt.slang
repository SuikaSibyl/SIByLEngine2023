#include "spt_interface.hlsli"
#include "../include/common/random.hlsli"
#include "../include/common/sampling.hlsli"
#include "../include/scene_descriptor_set.hlsli"
#include "../include/raytracer_descriptor_set.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
    } pConst;
};

[[vk::binding(0, 2)]]
RWTexture2D<float4> u_Color;

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

    // cosineSampleHemisphere
    // GeometryInfo geometry = geometries[payload.hit.geometryID];
    // MaterialInfo material = materials[geometry.materialID];
    
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
    return radiance;
}

[shader("raygeneration")]
void UDPTRgen() {
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