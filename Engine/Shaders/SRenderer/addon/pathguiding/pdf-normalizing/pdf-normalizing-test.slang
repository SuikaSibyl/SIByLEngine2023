#include "../../../include/common/cylindrical.hlsli"
#include "../../../include/common/light_impl.hlsli"
#include "../../../include/common/random.hlsli"
#include "../../../include/common/sampling.hlsli"
#include "../../../include/raytracer_descriptor_set.hlsli"
#include "../../../include/scene_descriptor_set.hlsli"
#include "../../../raytracer/spt_interface.hlsli"
#include "../../vbuffer/vbuffer.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    int2 debugPixel;
};

const Texture2D<uint4> u_vBuffer;
RWTexture2D<float> u_PdfNormalizing;

[shader("raygeneration")]
void RgenMain() {
    // load the first hit from the vbuffer
    const int2 pixel = int2(DispatchRaysIndex().xy);
    const VBufferData vhit = UnpackVBufferData(u_vBuffer[debugPixel]);
    if (IsInvalid(vhit)) {
        u_PdfNormalizing[pixel] = 0.f; // error color
        return;
    }

    const float2 uv = (float2(pixel) + 0.5f) / float2(512);
    const float3 dir = CylindricalToUnitVector(UnitToCylindrical(uv));
    
    RandomSamplerState RNG = InitRandomSampler(pixel, 0);
    const Ray primaryRay = generateRay(
        debugPixel,
        int2(resolution),
        false,
        globalUniform.cameraData);
    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);
    const Ray secondaryRay = SpawnRay(
        hit,
        dir);
    PrimaryPayload payload;
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xff, 0, 0, 0, ToRayDesc(secondaryRay), payload);
    if (HasHit(payload.hit) == false || dot(secondaryRay.direction, hit.geometryNormal) < 0.f) {
        u_PdfNormalizing[pixel] = 0.f; // error color
    }
    else {
        const PolymorphicLightInfo light = lights[0];
        const float3 color = EvaluateDirectLight(secondaryRay, payload.hit, light, RNG);
        const float irradiance = luminance(color);
        u_PdfNormalizing[pixel] = irradiance;
    }
}