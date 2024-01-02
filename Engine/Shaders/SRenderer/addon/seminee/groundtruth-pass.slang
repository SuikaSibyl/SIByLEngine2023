#include "../../raytracer/spt_interface.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/light_impl.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"

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
    float3 di = float3(0, 0, 0);
    float3 gi = float3(0, 0, 0);

    if (HasHit(payload.hit) == false)
        return float3(0, 0, 0);
    else if (pConst.renderMode == 0) {
        const PolymorphicLightInfo light = lights[0];
        // di += EvaluateDirectLight(primaryRay, payload.hit, light, RNG);
        const LightSample lightSample = SampleLight(payload.hit, light);
        Ray shadowRay = SpawnRay(payload.hit, lightSample.wi);
        shadowRay.tMax = distance(lightSample.position, payload.hit.position) - 0.01;
        const bool occluded = TraceOccludeRay(shadowRay, RNG, SceneBVH);
        const float visibility = occluded ? 0.0f : 1.0f;
        const float3 bsdf = EvalBsdf(payload.hit, -primaryRay.direction, lightSample.wi);

        const uint materialID = geometries[payload.hit.geometryID].materialID;
        const MaterialInfo material = materials[materialID];
        const float3 emission = material.emissiveColor;

        return lightSample.radiance * bsdf / lightSample.pdf + emission;

        // return shadowRay.origin * 0.2;
    }
    else if (pConst.renderMode == 2 || pConst.renderMode == 3) {
        const PolymorphicLightInfo light = lights[0];
        di += EvaluateDirectLight(primaryRay, payload.hit, light, RNG);
    }

    float3 multi_bounce = float3(0);
    GeometryHit primalHit = payload.hit;
    for (int i = 0; i < pConst.spp; ++i) {
        payload.hit = primalHit;
        // compute first bounce
        float bsdf_pdf;
        Ray bsdfRay = SpawnBsdfRay(payload.hit, -primaryRay.direction, RNG, bsdf_pdf);
        float3 first_bsdf = EvalBsdf(payload.hit, -primaryRay.direction, bsdfRay.direction);
        float3 throughput = first_bsdf / bsdf_pdf;
        if (bsdf_pdf == 0) throughput = float3(0, 0, 0);
        Intersection(bsdfRay, SceneBVH, payload, RNG);
        if (HasHit(payload.hit)) {
            const PolymorphicLightInfo light = lights[0];
            gi += EvaluateDirectLight(bsdfRay, payload.hit, light, RNG) * throughput;
            // further bounces
            multi_bounce += EvaluateMultibounceIndirect(bsdfRay, pConst.bounce_count, throughput, payload, RNG);
        }
    }
    for (int i = 0; i < pConst.one_bounce_spp; ++i) {
        payload.hit = primalHit;
        // compute first bounce
        float bsdf_pdf;
        Ray bsdfRay = SpawnBsdfRay(payload.hit, -primaryRay.direction, RNG, bsdf_pdf);
        float3 first_bsdf = EvalBsdf(payload.hit, -primaryRay.direction, bsdfRay.direction);
        float3 throughput = first_bsdf / bsdf_pdf;
        if (bsdf_pdf == 0) throughput = float3(0, 0, 0);
        Intersection(bsdfRay, SceneBVH, payload, RNG);
        if (HasHit(payload.hit)) {
            const PolymorphicLightInfo light = lights[0];
            gi += EvaluateDirectLight(bsdfRay, payload.hit, light, RNG) * throughput;
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
        
    float3 pixelColor = unidirectional_path_tracing(primaryRay, RNG);
    u_Color[pixel] = float4(pixelColor, 1);
}