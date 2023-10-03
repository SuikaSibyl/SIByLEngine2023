
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/common/light_impl.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../vbuffer/vbuffer.hlsli"

#include "../../include/common/vmf.hlsli"
#include "include/pathguiding.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
        uint strategy;
        bool adaption;
    } pConst;
};

Texture2D<uint4> u_vBuffer;

RWTexture2D<float4> u_vMFStatistics;
RWTexture2D<uint> u_epochCounter;
RWTexture2D<float4> u_color;

[shader("raygeneration")]
void RgenMain() {
    // Initialize stage
    // --------------------------------------------------------------------
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(pConst.resolution);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) {
        return;
    }
    RandomSamplerState RNG = InitRandomSampler(pixel, pConst.sample_batch);

    // Load Primary Hit from VBuffer
    // --------------------------------------------------------------------
    const Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );
    const VBufferData vhit = UnpackVBufferData(u_vBuffer[pixel]);
    if (IsInvalid(vhit)) {
        u_color[pixel] = float4(float3(0), 1);
        return;
    }
    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);
    float3 radiance = float3(0);
    PrimaryPayload payload;

    // BSDF importance sampling
    // --------------------------------------------------------------------
    float3 vplColor = float3(0);
    float3 bsdfRadiance = float3(0);
    
    float bsdf_pdf;
    Ray bsdfRay = SpawnBsdfRay(hit, -primaryRay.direction, RNG, bsdf_pdf);
    bsdf_pdf = PdfBsdfSample(hit, -primaryRay.direction, bsdfRay.direction);
    const float3 bsdf_dir = bsdfRay.direction;
    if (bsdf_pdf != 0) {
        Intersection(bsdfRay, SceneBVH, payload, RNG);
        if (HasHit(payload.hit)) {
            const PolymorphicLightInfo light = lights[0];
            vplColor = EvaluateDirectLight(bsdfRay, payload.hit, light, RNG);
            float3 first_bsdf = EvalBsdf(hit, -primaryRay.direction, bsdfRay.direction);
            float3 throughput = first_bsdf / bsdf_pdf;
            bsdfRadiance = vplColor * throughput;
        }
    }
    const float irradiance = luminance(vplColor);


    // Load the vMF statistics
    // --------------------------------------------------------------------
    vMFMLEStatistics vMFStat = vMFMLEStatistics(u_vMFStatistics[pixel]);
 
    // Guided sampling
    // --------------------------------------------------------------------
    vMFDistribution vMF = vMFDistribution(vMFStat);
    const float3 guidedDir = vMF.DrawSample(float2(GetNextRandom(RNG), GetNextRandom(RNG)));
    const float guidedPdf = vMF.Pdf(guidedDir);
    if (guidedPdf > 0) {
        Ray guideRay = SpawnRay(hit, guidedDir);
        payload.RND = RNG;
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xff, 0, 0, 0, ToRayDesc(guideRay), payload);
        RNG = payload.RND;
        // If we didn't hit anything, we're done
        if (HasHit(payload.hit) != false) {
            u_color[pixel] = float4(float3(0), 1);

            const PolymorphicLightInfo light = lights[0];
            const float3 color = EvaluateDirectLight(guideRay, payload.hit, light, RNG);

            float3 first_bsdf = EvalBsdf(hit, -primaryRay.direction, guideRay.direction);
            float3 throughput = first_bsdf / guidedPdf;

            float3 guided_radiance = color * throughput;
            float bsdfPdf = PdfBsdfSample(hit, -primaryRay.direction, guideRay.direction);
            
            if (bsdfPdf > 0) {
                if (pConst.strategy == 0) {
                    radiance += guided_radiance * guidedPdf / (guidedPdf + bsdfPdf);
                }
                else if (pConst.strategy == 1) {
                    radiance += guided_radiance;
                }
            }
        }
    }

    // Add the bsdf sample contribution
    // --------------------------------------------------------------------
    if (irradiance > 0) {
        const float guided_pdf = vMF.Pdf(bsdf_dir);
        if (pConst.strategy == 0) {
            radiance += bsdfRadiance * bsdf_pdf / (guided_pdf + bsdf_pdf);
        }
        else if (pConst.strategy == 2) {
            radiance += bsdfRadiance;
        }
    }

    // Update the vFM lobe statistics
    // --------------------------------------------------------------------
    if (irradiance > 0 && pConst.adaption) {
        uint epochCount = clamp(u_epochCounter[pixel], 0, 10);
        // exponential smoothing vMF
        const float alpha = pow(0.7, epochCount);
        const float sumWeights = irradiance / bsdf_pdf;
        const float3 sumeightedDirections = bsdfRay.direction * sumWeights;
        vMFStat.sumeightedDirections = ExponentialSmooth(vMFStat.sumeightedDirections, sumeightedDirections, alpha);
        vMFStat.sumWeights = ExponentialSmooth(vMFStat.sumWeights, sumWeights, alpha);
        // update vMF statistics & epoch counter
        u_vMFStatistics[pixel] = vMFStat.Pack();
        u_epochCounter[pixel] += 1;
    }

    u_color[pixel] = float4(radiance, 1);
}