#include "../../include/common/cylindrical.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "seminee-config.hlsli"
#include "seminee_interface.hlsli"
#include "seminee_internal.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
        int sample_hint;
        int2 target_pixel;
    } pConst;
};

RWTexture2D<float4> u_Target;
RWTexture2D<float4> u_Estimate;
// RWStructuredBuffer<VPLData> u_VPLData;
// RWStructuredBuffer<int4> u_SelectedVPL;

void DistribuitionVisualize(
    in_ref(int2) pixel,
    in_ref(float3) position,
    in_ref(float3) normal)
{
}

void TargetPlot(
    in_ref(int2) pixel,
    in_ref(GeometryHit) hit,
    in_ref(Ray) secondaryRay,
    inout_ref(RandomSamplerState) RNG)
{
    PrimaryPayload payload;
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xff, 0, 0, 0, ToRayDesc(secondaryRay), payload);
    if (HasHit(payload.hit) == false) {
        u_Target[pixel] = float4(0, 0, 0, 1);
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

        float dotNL = saturate(dot(hit.geometryNormal, secondaryRay.direction));
        float3 gi_luminance = albedo.xyz * lightColor * saturate(dot(payload.hit.geometryNormal, -lightDir)) * visibility;
        u_Target[pixel] = float4(float3(luminance(gi_luminance) * dotNL), 1);
        // u_Target[pixel] = float4(1, 0, 1, 1);
    }
}

[shader("raygeneration")]
void RaygenMain() {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(512);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    // If the pixel is outside of the image, don't do anything:
    if(any(pixel >= resolution)) {
        return;
    }
    RandomSamplerState RNG = InitRandomSampler(pixel, pConst.sample_batch);

    const float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    const float3 dir = CylindricalToUnitVector(UnitToCylindrical(uv));

    // Trace primary ray to target pixel
    Ray primaryRay = generateRay(
        pConst.target_pixel,
        int2(pConst.resolution),
        false,
        globalUniform.cameraData
    );
    RayDesc ray = ToRayDesc(primaryRay);
    PrimaryPayload payload;
    payload.RND = RNG;
    uint instanceMask = 0xff;
    uint rayFlags = RAY_FLAG_NONE;
    TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ray, payload);
    RNG = payload.RND;

    if (HasHit(payload.hit) == false) {
        u_Target[pixel] = float4(0, 0, 0, 1);
        u_Estimate[pixel] = float4(0, 0, 0, 1);
        return;
    }

    // Trace secondary ray
    Ray secondaryRay = SpawnRay(
        payload.hit,
        dir
    );

    TargetPlot(pixel, payload.hit, secondaryRay, RNG);

    // u_Target[pixel] = float4((dir + 1) / 2, 1);
    u_Estimate[pixel] = float4((dir + 1) / 2, 1);

    // DistribuitionVisualize(pixel, payload.hit.position, payload.hit.geometryNormal);
}