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
    float3 di = float3(0, 0, 0);
    float3 gi = float3(0, 0, 0);

    if (HasHit(payload.hit) == false)
        return float3(0, 0, 0);
    else {
        const PolymorphicLightInfo light = lights[0];
        di += EvaluateDirectLight(primaryRay, payload.hit, light, RNG);
    }

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

    if (pConst.renderMode == 0) {
        return di;
    }
    else if (pConst.renderMode == 1) {
        return gi;
    }
    else {
        return di + gi;
    }
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