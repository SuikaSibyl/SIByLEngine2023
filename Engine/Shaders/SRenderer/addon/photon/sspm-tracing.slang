#include "../../include/common/light_impl.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int max_iteration_count;
    uint rand_seed;
    float roughness_threshold;
    float intenisty_threshold;
};

RWTexture2D<uint4> u_photon_map;

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

    // The photon's intensity is negligibly small
    if (luminance(radiance) < intenisty_threshold) {
        return false;
    }
    // The location to be stored is out of screen

    // The photon travels beyond the stored depth buffer values

    // The photon is part of direct lighting
    
    return true;
}

void StorePhonton(
    in_ref(GeometryHit) hit,
    in_ref(float3) radiance,
    out_ref(int2) pixel
) {
    InterlockedAdd(u_photon_map[pixel].x, 1u);
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
        if (!HasHit(payload.hit)) {
            break;
        }
        const MaterialInfo material = materials[geometries[payload.hit.geometryID].materialID];
        // If should store, then store a photon
        int2 pixel;
        if (prev_is_non_diffuse && ShouldStorePhoton(payload.hit, material, radiance, pixel)) {
            StorePhonton(payload.hit, radiance, pixel);
        }
        // Check whether the current bounce is diffuse bounce
        if (material.bsdfID != 0 &&
            material.roughness < roughness_threshold) {
            prev_is_non_diffuse = true;
        } else {
            prev_is_non_diffuse = false;
        }
    }
}