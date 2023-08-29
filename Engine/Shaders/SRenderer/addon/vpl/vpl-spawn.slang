#include "../../raytracer/spt_interface.hlsli"
#include "../../raytracer/lights/directional.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/common/light_impl.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        float4 bounding_sphere;
        uint sample_batch;
        int max_depth;
    } pConst;
};

RWStructuredBuffer<uint> u_vplCounter;
RWStructuredBuffer<float4> u_vplPositions;
RWStructuredBuffer<float4> u_vplNormals;
RWStructuredBuffer<float4> u_vplColors;

[shader("raygeneration")]
void RgenMain() {
    // Initialize thread RNG
    if (any(DispatchRaysIndex().xy >= DispatchRaysDimensions().xy)) return;
    RandomSamplerState RNG = InitRandomSampler(DispatchRaysIndex().xy, pConst.sample_batch);

    LightSampleInfo info;
    info.boundingSphere = pConst.bounding_sphere;

    // Selected light
    const PolymorphicLightInfo light = lights[0];
    LightRaySample lightRaySamp = SampleLightRay(light, info, RNG);    
    Ray lightRay = lightRaySamp.ray;
    
    PrimaryPayload payload;

    // Should be similar to the radiance of the chosen light sample
    float3 throughput = lightRaySamp.radiance / (lightRaySamp.pdf_position * lightRaySamp.pdf_direction);
    for (int depth = 0; depth < pConst.max_depth; ++depth) {
        // Trace the light ray
        payload.RND = RNG;
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xff, 0, 0, 0, ToRayDesc(lightRay), payload);
        RNG = payload.RND;

        // If we didn't hit anything, we're done
        if(HasHit(payload.hit) == false)
            break;

        // update the throughput
        const GeometryInfo geometry = geometries[payload.hit.geometryID];
        const MaterialInfo material = materials[geometry.materialID];
        const float4 albedo = textures[material.baseOrDiffuseTextureIndex].Sample(payload.hit.texcoord);
        throughput *= albedo.xyz / k_pi;
        
        // Store the VPL
        uint VPLid;
        InterlockedAdd(u_vplCounter[0], 1, VPLid);
        u_vplPositions[VPLid] = float4(payload.hit.position, 0.0);
        u_vplNormals[VPLid] = float4(payload.hit.geometryNormal, 0.0);
        u_vplColors[VPLid] = float4(throughput, 0.0);
        
        // Quit if we reach max depth
        if (depth + 1 >= pConst.max_depth)
            break;
        // Sample the next light ray if we need further bounces
        const float2 uv = float2(GetNextRandom(RNG), GetNextRandom(RNG));
        lightRay = SpawnRay(payload.hit, CosineWeightedHemisphereSample(payload.hit.geometryNormal, uv));
    }
}