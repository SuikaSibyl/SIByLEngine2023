#include "../../include/common/cylindrical.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/common/vmf.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../vbuffer/vbuffer.hlsli"
#include "include/pathguiding.hlsli"


[[vk::push_constant]]
cbuffer PushConstants {
    int2 resolution;
    int2 debugPixel;
    int mode;
    float scalar;
};

Texture2D<float4> u_vMFStatistics;
Texture2D<uint4> u_vBuffer;

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

    const float2 uv = (float2(pixel) + 0.5f) / float2(512);
    const float3 dir = CylindricalToUnitVector(UniformToCylindrical(uv));

    const vMFMLEStatistics vMFS = vMFMLEStatistics(u_vMFStatistics[debugPixel]);
    const vMFDistribution vMF = vMFDistribution(vMFS);
    const float importance = vMF.Pdf(dir);
    
    if (mode == 0) {
        u_output[pixel] = float4(float3(importance * scalar), 1);
    }
    else {
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
        if (HasHit(payload.hit) == false) {
            u_output[pixel] = float4(importance * scalar, 0, 0, 1);
        }
        else {
            const PolymorphicLightInfo light = lights[0];
            float3 lightColor = unpackLightColor(light);
            float3 lightDir = Unorm32OctahedronToUnitVector(light.databyte0);

            Ray shadowRay = SpawnRay(payload.hit, -lightDir);
            bool occluded = TraceOccludeRay(shadowRay, RNG, SceneBVH);
            float visibility = occluded ? 0.0f : 1.0f;

            float3 second_bsdf = EvalBsdf(payload.hit, -secondaryRay.direction, -lightDir);

            const float3 color = lightColor * second_bsdf * visibility;
            const float irradiance = luminance(color);
            u_output[pixel] = float4(importance * scalar, irradiance, 0, 1);
        }
    }
}