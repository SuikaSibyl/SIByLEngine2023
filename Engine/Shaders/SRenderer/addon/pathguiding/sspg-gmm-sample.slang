#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/common/light_impl.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../include/common/cylindrical.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../vbuffer/vbuffer.hlsli"

#include "include/gaussian_mm.hlsli"
#include "include/pathguiding.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
        uint strategy;
        bool adaption;
        float exponential_factor;
    } pConst;
};

Texture2D<uint4> u_vBuffer;

RWTexture2D<float4> u_gmmStatisticsPack0;
RWTexture2D<float4> u_gmmStatisticsPack1;
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
    const float2 cylcoord = CylindricalToUnit(UnitVectorToCylindrical(bsdf_dir));
    
    // Load the GMM statistics
    // --------------------------------------------------------------------
    const float4 pack0 = u_gmmStatisticsPack0[pixel];
    const float4 pack1 = u_gmmStatisticsPack1[pixel];
    uint epoch_count = uint(pack1.z);
    GMMStatictics GMMstat = UnpackStatistics(pack0, pack1);
    
    // Guided sampling
    // --------------------------------------------------------------------
    // const MultivariateGaussian2D GMM = CreateDistribution(GMMstat);
    MultivariateGaussian2D GMM = MultivariateGaussian2D(float2(GMMstat.ex, GMMstat.ey), float2x2(0.01, 0, 0, 0.01));
    const float2 GMMsample = GMM.DrawSample(float2(GetNextRandom(RNG), GetNextRandom(RNG)));
    const float3 guidedDir = CylindricalToUnitVector(UnitToCylindrical(GMMsample));
    const float guidedPdf = GMM.Pdf(GMMsample) / float(4 * k_pi);

    if (guidedPdf > 0 && all(GMMsample >= float2(0) && GMMsample <= float2(1))) {
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
            float3 throughput = float3(first_bsdf / guidedPdf);

            float3 guided_radiance = color * throughput;
            float bsdfPdf = PdfBsdfSample(hit, -primaryRay.direction, guideRay.direction);
            
            if (bsdfPdf > 0) {
                if (pConst.strategy == 0) {
                    radiance += float3(guided_radiance * guidedPdf / (guidedPdf + bsdfPdf));
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
        const float guided_pdf = GMM.Pdf(cylcoord) / float(4 * k_pi);
        if (pConst.strategy == 0) {
            if (guided_pdf > 0) {
                radiance += float3(bsdfRadiance * bsdf_pdf / (guided_pdf + bsdf_pdf));
            } else {
                radiance += bsdfRadiance;
            }
        }
        else if (pConst.strategy == 2) {
            radiance += bsdfRadiance;
        }
    }
    
    // Update the vFM lobe statistics
    // --------------------------------------------------------------------
    if (irradiance > 0 && pConst.adaption) {
        uint epochCount = clamp(epoch_count, 0, 10);
        // exponential smoothing vMF
        const float alpha = pow(pConst.exponential_factor, epochCount);
        // const float sumWeights = irradiance / bsdf_pdf;
        const float sumWeights = 1;
        
        const float4 new_pack0 = sumWeights * float4(cylcoord.x, cylcoord.y, cylcoord.x * cylcoord.x, cylcoord.y * cylcoord.y);
        const float2 new_pack1 = sumWeights * float2(cylcoord.x * cylcoord.y, 1);
        const float4 update_pack0 = ExponentialSmooth(pack0, new_pack0, alpha);
        const float2 update_pack1 = ExponentialSmooth(pack1.xy, new_pack1, alpha);
        epoch_count += 1;
        
        u_gmmStatisticsPack0[pixel] = update_pack0;
        u_gmmStatisticsPack1[pixel] = float4(update_pack1, epoch_count, 0);
    }

    // // radiance += material.emissiveColor;
    // const PolymorphicLightInfo light = lights[0];
    // radiance += EvaluateDirectLight(primaryRay, hit, light, RNG);
    
    u_color[pixel] = float4(radiance, 1);
}