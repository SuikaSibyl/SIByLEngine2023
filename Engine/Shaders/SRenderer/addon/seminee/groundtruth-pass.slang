#include "../../raytracer/spt_interface.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/light_impl.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../vbuffer/vbuffer.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
        int renderMode;
        int bounce_count;
        int spp;
        int one_bounce_spp;
    } pConst;
};

RWTexture2D<float4> u_Color;
Texture2D<uint4> u_VBuffer;

float3 unidirectional_path_tracing(in_ref(Ray) primaryRay, inout_ref(RandomSamplerState) RNG, VBufferData vhit) {
    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);
    GeometryInfo geometry = geometries[hit.geometryID];
    MaterialInfo material = materials[geometry.materialID];

    PrimaryPayload payload;
    payload.RND = RNG;

    const PolymorphicLightInfo light = lights[0];
    float3 di = EvaluateDirectLight(primaryRay, hit, light, RNG);

    if (pConst.renderMode == 0) {
        return di;
    }

    float3 gi = float3(0);
    float3 multi_bounce = float3(0);

    float3 bsdfRadiance = float3(0);
    float3 first_gi = float3(0);

    for (int i = 0; i < pConst.spp; ++i) {
        float3 throughput = float3(1); float bsdf_pdf;
        Ray bsdfRay = SpawnBsdfRay(hit, -primaryRay.direction, RNG, bsdf_pdf);
        float3 first_bsdf = EvalBsdf(hit, -primaryRay.direction, bsdfRay.direction);
        throughput = first_bsdf / bsdf_pdf;
        float3 vpl_position = float3(0);

        float3 extra_gi = float3(0);
        const float3 bsdf_dir = bsdfRay.direction;
        if (bsdf_pdf != 0) {
            Intersection(bsdfRay, SceneBVH, payload, RNG);
            if (HasHit(payload.hit)) {
                const PolymorphicLightInfo light = lights[0];
                bsdfRadiance = EvaluateDirectLight(bsdfRay, payload.hit, light, RNG) * throughput;
                gi += bsdfRadiance;
                multi_bounce += EvaluateMultibounceIndirect(bsdfRay, 1, throughput, payload, RNG);
            }
        }
    }
    for (int i = 0; i < pConst.one_bounce_spp; ++i) {
        float3 throughput = float3(1); float bsdf_pdf;
        Ray bsdfRay = SpawnBsdfRay(hit, -primaryRay.direction, RNG, bsdf_pdf);
        if (bsdf_pdf != 0) {
            Intersection(bsdfRay, SceneBVH, payload, RNG);
            if (HasHit(payload.hit)) {
                const PolymorphicLightInfo light = lights[0];
                bsdfRadiance = EvaluateDirectLight(bsdfRay, payload.hit, light, RNG) * throughput;
                gi += bsdfRadiance;
            }
        }
    }

    gi /= (pConst.spp + pConst.one_bounce_spp);
    multi_bounce /= pConst.spp;

    if (pConst.renderMode == 0) {
        return di;
    }
    else if (pConst.renderMode == 1) {
        // if (HasHit(payload.hit)) {
        //     const PolymorphicLightInfo light = lights[0];
        //     gi += EvaluateDirectLight(bsdfRay, payload.hit, light, RNG) * throughput;
        // }
        return gi;
    }
    else if (pConst.renderMode == 2) {
        return gi + multi_bounce;
    }
    else if (pConst.renderMode == 3) {
        return gi + multi_bounce + di;
    }
    return float3(0, 0, 0);
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
    // const float2 jitter_pixel = pixel + GetNextRandomFloat2(RNG) - float2(0.5); // manually jitter

    Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );
    const VBufferData vhit = UnpackVBufferData(u_VBuffer[pixel]);
    if (IsInvalid(vhit)) {
        u_Color[pixel] = float4(0, 0, 0, 1);
        return;
    }

    float3 pixelColor = unidirectional_path_tracing(primaryRay, RNG, vhit);
    u_Color[pixel] = float4(pixelColor, 1);
}