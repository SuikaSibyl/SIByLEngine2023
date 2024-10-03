#include "common/gaussian.hlsli"
#include "common/math.hlsli"
#include "srenderer/lights.hlsli"
#include "srenderer/lights/envmap.hlsli"
#include "srenderer/materials/lambertian.hlsli"
#include "srenderer/rayquery.hlsli"
#include "srenderer/materials.hlsli"

[[vk::push_constant]] cbuffer PushConstants {
    int random_seed;
    InspectorMode inspector_mode;
};

enum InspectorMode {
    ALBEDO,
    EMISSION,
    FACEFORWARD,
};

RWTexture2D<float4> output;

float3 inspector_pt(
    int2 pixel, int2 resolution, 
    inout RandomSamplerState RNG
) {
    // cast primary ray
    // const float2 jitter = GetNextRandomFloat2(RNG) - 0.5f;
    const float2 jitter = 0;
    Ray ray = generateRay(pixel + jitter, resolution, false, fetchActiveCamera());

    PrimaryPayload payload = PrimaryRayQuery(ray);
    bool hit = HasHit(payload.hit);
    
    // Show
    if (hit) {
        MaterialData mat = fetchMaterialData(payload.hit);
        switch (inspector_mode) {
        case InspectorMode::ALBEDO: return materials::albedo(mat); break;
        case InspectorMode::EMISSION: return materials::emission(mat); break;
        case InspectorMode::FACEFORWARD: return IsFaceForward(payload.hit) ? float3(0, 1, 0) : float3(1, 0, 0); break;
        }
    }

    return float3(1, 0, 1);
}

[shader("compute")]
[numthreads(32, 4, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(1024, 1024);
    const int2 pixel = int2(dtid.xy);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;

    RandomSamplerState RNG = InitRandomSampler(dtid.xy, random_seed);
    output[pixel] = float4(inspector_pt(pixel, resolution, RNG), 1);
}