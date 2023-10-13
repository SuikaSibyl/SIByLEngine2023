#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/common/light_impl.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../include/common/cylindrical.hlsli"
#include "../../include/common/concentric_map.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../gbuffer/gbuffer_interface.hlsli"
#include "../gbuffer/gbuffer_prev_interface.hlsli"

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
        bool extra_bounce;
        int spp;
        bool extra_half_spp;
        bool learn_first;
    } pConst;
};

RWTexture2D<float4> u_gmmStatisticsPack0;
RWTexture2D<float4> u_gmmStatisticsPack1;
RWTexture2D<float4> u_color;
RWTexture2D<float4> u_vpls;

const Texture2D<float4> u_gmmStatisticsPack0Prev;
const Texture2D<float4> u_gmmStatisticsPack1Prev;

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

    // Load Primary surface from GBuffer
    // --------------------------------------------------------------------
    const Ray primaryRay = generateRay(pixel, resolution, false, globalUniform.cameraData);
    const ShadingSurface surface = GetGBufferSurface(pixel, globalUniform.cameraData);
    // Load the previous GBuffer surface
    const float3 motionVector = GetMotionVector(pixel);
    const int2 pixel_prev = int2(round(float2(pixel) + motionVector.xy));
    const ShadingSurface temporalSurface = GetPrevGBufferSurface(pixel_prev, prevGlobalUniform.cameraData);
    const float expectedPrevLinearDepth = surface.viewDepth + motionVector.z;
    float3x3 frame = createFrame(surface.shadingNormal);

    // If the surface is invalid, we're done
    if (!IsShadingSurfaceValid(surface)) {
        u_color[pixel] = float4(float3(0), 1);
        u_gmmStatisticsPack0[pixel] = float4(0);
        u_gmmStatisticsPack1[pixel] = float4(0);
        return;
    }

    // Otherwise, let's do some shading!
    float3 radiance = float3(0);
    PrimaryPayload payload;

    float extra_half = pConst.extra_half_spp ? 2 : 1;

    // Load the GMM statistics
    // --------------------------------------------------------------------
    float4 pack0 = float4(0);
    float4 pack1 = float4(0);
    {   // constant params
        const float depthThreshold = 0.1;
        const float normalThreshold = 0.5;
        // skip the sample if the surface is invalid
        if (IsShadingSurfaceValid(temporalSurface) &&
            IsValidNeighbor(surface.geometryNormal, temporalSurface.geometryNormal,
                            expectedPrevLinearDepth, temporalSurface.viewDepth,
                            normalThreshold, depthThreshold) &&
            AreMaterialsSimilar(surface, temporalSurface)
        ) {
            pack0 = u_gmmStatisticsPack0Prev[pixel_prev];
            pack1 = u_gmmStatisticsPack1Prev[pixel_prev];
        }
    }
    uint epoch_count = uint(pack1.z);
    GMMStatictics GMMstat = UnpackStatistics(pack0, pack1);
    MultivariateGaussian2D GMM = MultivariateGaussian2D(float2(GMMstat.ex, GMMstat.ey), float2x2(0.01, 0, 0, 0.01));
   
    // BSDF importance sampling
    // --------------------------------------------------------------------
    float3 bsdfRadiance = float3(0);
    float3 first_gi = float3(0);
    float bsdf_pdf;
    Ray bsdfRay = SpawnBsdfRay(surface, -primaryRay.direction, RNG, bsdf_pdf);
    float3 first_bsdf = EvalBsdf(surface, -primaryRay.direction, bsdfRay.direction);
    float3 throughput = first_bsdf / bsdf_pdf;
    float3 vpl_position = float3(0);

    float3 extra_gi = float3(0);
    const float3 bsdf_dir = bsdfRay.direction;
    if (bsdf_pdf != 0) {
        Intersection(bsdfRay, SceneBVH, payload, RNG);
        if (HasHit(payload.hit)) {
            const PolymorphicLightInfo light = lights[0];
            bsdfRadiance = EvaluateDirectLight(bsdfRay, payload.hit, light, RNG) * throughput;
            vpl_position = payload.hit.position;
            first_gi = bsdfRadiance;
            if (pConst.extra_bounce) {
                extra_gi = EvaluateMultibounceIndirect(bsdfRay, 1, throughput, payload, RNG);
                bsdfRadiance += extra_gi;
            }
        }
    }
    // Add the bsdf sample contribution
    const float2 square_coord = FromConcentricMap(UniformHemisphereToConcentricDisk(to_local(frame, bsdf_dir)));
    const float irradiance = luminance(bsdfRadiance);
    if (irradiance > 0) {
        const float guided_pdf = GMM.Pdf(square_coord) / float(2 * k_pi);
        if (pConst.strategy == 0) {
            if (guided_pdf > 0) {
                radiance += float3(first_gi * bsdf_pdf / (extra_half * guided_pdf + bsdf_pdf));
                radiance += float3(extra_gi * bsdf_pdf / (guided_pdf + bsdf_pdf));
            } else {
                radiance += bsdfRadiance;
            }
        }
        else if (pConst.strategy == 2) {
            radiance += bsdfRadiance;
        }
    }

    // Guided sampling
    // --------------------------------------------------------------------
    const float2 GMMsample = GMM.DrawSample(float2(GetNextRandom(RNG), GetNextRandom(RNG)));
    // const float3 guidedDir = CylindricalToUnitVector(UnitToCylindrical(GMMsample));
    const float3 guidedDir = to_world(frame, ConcentricDiskToUniformHemisphere(ToConcentricMap(GMMsample)));
    const float guidedPdf = GMM.Pdf(GMMsample) / float(2 * k_pi);

    if (guidedPdf > 0 && all(GMMsample >= float2(0) && GMMsample <= float2(1))) {
        Ray guideRay = SpawnRay(surface, guidedDir);
        payload.RND = RNG;
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xff, 0, 0, 0, ToRayDesc(guideRay), payload);
        RNG = payload.RND;
        // If we didn't hit anything, we're done
        if (HasHit(payload.hit) != false) {
            float3 first_bsdf = EvalBsdf(surface, -primaryRay.direction, guideRay.direction);
            float3 throughput = float3(first_bsdf / guidedPdf);
            const PolymorphicLightInfo light = lights[0];
            float3 color = EvaluateDirectLight(guideRay, payload.hit, light, RNG) * throughput;
            float3 indirect_color = float3(0);
            if (pConst.extra_bounce) {
                indirect_color += EvaluateMultibounceIndirect(guideRay, 1, throughput, payload, RNG);
            }

            float bsdfPdf = PdfBsdfSample(surface, -primaryRay.direction, guideRay.direction);

            if (bsdfPdf > 0) {
                if (pConst.strategy == 0) {
                    radiance += float3(color * guidedPdf / (extra_half * guidedPdf + bsdfPdf));
                    // radiance += float3(color * guidedPdf / (guidedPdf + bsdfPdf));
                    radiance += float3(indirect_color * guidedPdf / (guidedPdf + bsdfPdf));
                }
                else if (pConst.strategy == 1) {
                    radiance += color + indirect_color;
                }
            }
        }
    }

    if (pConst.extra_half_spp) {
        // extra
        const float2 GMMsample = GMM.DrawSample(float2(GetNextRandom(RNG), GetNextRandom(RNG)));
        const float3 guidedDir = to_world(frame, ConcentricDiskToUniformHemisphere(ToConcentricMap(GMMsample)));
        const float guidedPdf = GMM.Pdf(GMMsample) / float(2 * k_pi);

        if (guidedPdf > 0 && all(GMMsample >= float2(0) && GMMsample <= float2(1))) {
            Ray guideRay = SpawnRay(surface, guidedDir);
            payload.RND = RNG;
            TraceRay(SceneBVH, RAY_FLAG_NONE, 0xff, 0, 0, 0, ToRayDesc(guideRay), payload);
            RNG = payload.RND;
            // If we didn't hit anything, we're done
            if (HasHit(payload.hit) != false) {
                float3 first_bsdf = EvalBsdf(surface, -primaryRay.direction, guideRay.direction);
                float3 throughput = float3(first_bsdf / guidedPdf);
                const PolymorphicLightInfo light = lights[0];
                float3 color = EvaluateDirectLight(guideRay, payload.hit, light, RNG) * throughput;
                float bsdfPdf = PdfBsdfSample(surface, -primaryRay.direction, guideRay.direction);
                
                if (bsdfPdf > 0) {
                    if (pConst.strategy == 0) {
                        radiance += float3(color * guidedPdf / (extra_half * guidedPdf + bsdfPdf));
                    }
                }
            }
        }
    }

    if (pConst.learn_first) bsdfRadiance = first_gi;
    const float sumweight = (bsdf_pdf != 0) ? luminance(bsdfRadiance) / bsdf_pdf : 0;
    u_vpls[pixel] = float4(vpl_position, sumweight);
    
    u_gmmStatisticsPack0[pixel] = pack0;
    u_gmmStatisticsPack1[pixel] = pack1;

    // radiance += material.emissiveColor;
    const PolymorphicLightInfo light = lights[0];
    radiance += EvaluateDirectLight(primaryRay, surface, light, RNG);
    
    u_color[pixel] = float4(radiance, 1);
}