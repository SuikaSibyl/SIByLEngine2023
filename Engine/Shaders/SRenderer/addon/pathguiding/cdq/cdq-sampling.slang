#include "../../../include/common/cylindrical.hlsli"
#include "../../../include/common/light_impl.hlsli"
#include "../../../include/common/random.hlsli"
#include "../../../include/common/sampling.hlsli"
#include "../../../include/common/vmf.hlsli"
#include "../../../include/raytracer_descriptor_set.hlsli"
#include "../../../include/scene_descriptor_set.hlsli"
#include "../../../raytracer/spt_interface.hlsli"
#include "../../vbuffer/vbuffer.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    int2 debugPixel;
    int rand_seed;
};

const Texture2D<uint4> u_vBuffer;
RWTexture2D<float4> u_PreSampleList;

[shader("raygeneration")]
void RgenMain() {
    // load the first hit from the vbuffer
    const int2 pixel = int2(DispatchRaysIndex().xy);
    const VBufferData vhit = UnpackVBufferData(u_vBuffer[debugPixel]);
    if (IsInvalid(vhit)) {
        u_PreSampleList[pixel] = float4(0, 0, 0, 0); // error color
        return;
    }
    
    const Ray primaryRay = generateRay(
        debugPixel,
        int2(resolution),
        false,
        globalUniform.cameraData);
    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);
    RandomSamplerState RNG = InitRandomSampler(pixel, rand_seed);

    float bsdf_pdf;
    Ray bsdfRay = SpawnBsdfRay(hit, -primaryRay.direction, RNG, bsdf_pdf);
    PrimaryPayload payload;
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xff, 0, 0, 0, ToRayDesc(bsdfRay), payload);
    if (HasHit(payload.hit) == false) {
        u_PreSampleList[pixel] = float4(bsdfRay.direction, 0);
    }
    else {
        const PolymorphicLightInfo light = lights[0];
        const float3 color = EvaluateDirectLight(bsdfRay, payload.hit, light, RNG);
        const float irradiance = luminance(color);
        u_PreSampleList[pixel] = float4(bsdfRay.direction, irradiance);
    }
}