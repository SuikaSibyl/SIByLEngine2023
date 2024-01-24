#include "../../include/common/light_impl.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "common.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    int2 photon_map_size;
    float roughness_threshold;
    float intenisty_threshold;
    int max_iteration_count;
    uint rand_seed;
};

RWByteAddressBuffer u_photon_map;
Texture2D<float2> u_depth;

void StorePhonton(
    in_ref(GeometryHit) hit,
    in_ref(float3) radiance,
    in_ref(int2) pixel,
) {
    const uint flatten = FlattensPixelToIndex(pixel, resolution.x);
    InterloeckedAddFloat3(u_photon_map, flatten, radiance);
}

/**
 * Screen-Space Photon Mapping
 * Emission pass to spawn photons from light sources
 */

bool ShouldStorePhoton(
    in_ref(GeometryHit) hit,
    in_ref(MaterialInfo) material,
    in_ref(float3) radiance,
    out_ref(int2) pixel
) {
    // If the photon is not a caustics photon
    // We should not store it

    const float3 posPS = WS2PS(hit.position, globalUniform.cameraData.viewProjMat);
    const float depth_pos = distance(hit.position, globalUniform.cameraData.posW);
    pixel = int2(posPS.xy * resolution);

    // The photon's intensity is negligibly small
    if (luminance(radiance) <= intenisty_threshold) return false;
    // The location to be stored is out of screen
    if (any(pixel < 0) || any(pixel >= resolution)) return false;
    // The photon travels beyond the stored depth buffer values
    const float2 depth = u_depth[pixel];
    if (depth_pos >= depth.x) return false;
    // The photon is part of direct lighting
    return true;
}

[shader("raygeneration")]
void RgenMain() {
    const int2 gid = int2(DispatchRaysIndex().xy);
    RandomSamplerState RNG = InitRandomSampler(gid, rand_seed);

    // Spawn a photon from a light source
    LightSampleInfo light_info;
    const PolymorphicLightInfo light = lights[0];
    LightRaySample light_sample = SampleLightRay(light, light_info, RNG);
    PrimaryPayload payload;
    Ray light_ray = light_sample.ray;
    bool prev_is_non_diffuse = false;

    // Photon tracing
    float3 radiance = light_sample.radiance;
    for (int i = 0; i < max_iteration_count; ++i) {
        Intersection(light_ray, SceneBVH, payload, RNG);
        if (!HasHit(payload.hit)) break;
        
        const MaterialInfo material = materials[geometries[payload.hit.geometryID].materialID];
        // If should store, then store a photon

        int2 pixel;
        if (prev_is_non_diffuse && ShouldStorePhoton(payload.hit, material, radiance, pixel)) {
            StorePhonton(payload.hit, radiance, pixel);
        }
        // Check whether the current bounce is diffuse bounce
        if (material.roughness <= roughness_threshold) {
            prev_is_non_diffuse = true;
        } else break;
        
        float bsdf_pdf;
        light_ray = SpawnBsdfRay(payload.hit, -light_ray.direction, RNG, bsdf_pdf);
        radiance /= bsdf_pdf;
    }
}