#include "../../include/common/concentric_map.hlsli"
#include "../../include/common/cylindrical.hlsli"
#include "../../include/common/light_impl.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/common/vmf.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../vbuffer/vbuffer.hlsli"
#include "include/gaussian_mm.hlsli"
#include "include/pathguiding.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    int2 debugPixel;
    int mode;
    float scalar;
};

const StructuredBuffer<float4> u_gmmStatisticsPack0;
const StructuredBuffer<float4> u_gmmStatisticsPack1;
Texture2D<uint4> u_vBuffer;
RWTexture2D<float> u_pdfUnormalized;
RWTexture2D<float4> u_output;

[shader("raygeneration")]
void RgenMain() {
    // load the first hit from the vbuffer
    const int2 pixel = int2(DispatchRaysIndex().xy);
    const VBufferData vhit = UnpackVBufferData(u_vBuffer[debugPixel]);
    if (IsInvalid(vhit)) {
        u_output[pixel] = float4(1, 0, 1, 1); // error color
        return;
    }
    RandomSamplerState RNG = InitRandomSampler(pixel, 0);
    const Ray primaryRay = generateRay(
        debugPixel,
        int2(resolution),
        false,
        globalUniform.cameraData);
    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);

    const float2 uv = (float2(pixel) + 0.5f) / float2(512);
    const float3 dir = CylindricalToUnitVector(UnitToCylindrical(uv));

    // Load the GMM statistics
    // --------------------------------------------------------------------
    GMM2D GMMs;
    const int index = (debugPixel.y * resolution.x + debugPixel.x) * 4;
    for (int i = 0; i < 4; ++i) {
        GMMs.sufficientStats0[i] = u_gmmStatisticsPack0[index + i];
        GMMs.sufficientStats1[i] = u_gmmStatisticsPack1[index + i];
    }
    GMMs.build();

    const float3x3 frame = createFrame(hit.shadingNormal);
    const float2 square_coord = FromConcentricMap(UniformHemisphereToConcentricDisk(to_local(frame, dir)));

    // GMM.inverse = float2x2(1, 0, 0, 1);
    // GMM.mean = float2(0.5, 0.5);
    // GMM.det = 1.f;
    const float importance = GMMs.Pdf(square_coord);

    if (mode == 0) {
        u_output[pixel] = float4(importance * scalar, 0, 0, 1);
        u_pdfUnormalized[pixel] = importance;
    }
    else {
        const Ray secondaryRay = SpawnRay(
            hit,
            dir);
        PrimaryPayload payload;
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xff, 0, 0, 0, ToRayDesc(secondaryRay), payload);
        if (HasHit(payload.hit) == false) {
            u_output[pixel] = float4(importance * scalar, 0, 0, 1);
        }
        else {
            const PolymorphicLightInfo light = lights[0];
            const float3 color = EvaluateDirectLight(secondaryRay, payload.hit, light, RNG);
            const float irradiance = luminance(color);
            u_output[pixel] = float4(importance * scalar, irradiance, 0, 1);
        }
    }
}