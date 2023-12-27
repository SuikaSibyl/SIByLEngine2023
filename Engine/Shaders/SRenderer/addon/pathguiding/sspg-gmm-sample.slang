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
        int lobeCount;
    } pConst;
};

RWTexture2D<float4> u_color;
RWTexture2D<float4> u_vpls;

RWStructuredBuffer<float4> u_gmmStatisticsPack0;
RWStructuredBuffer<float4> u_gmmStatisticsPack1;
const StructuredBuffer<float4> u_gmmStatisticsPack0Prev;
const StructuredBuffer<float4> u_gmmStatisticsPack1Prev;

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

    const int index = (pixel.y * resolution.x + pixel.x) * pConst.lobeCount;

    // If the surface is invalid, we're done
    if (!IsShadingSurfaceValid(surface)) {
        u_color[pixel] = float4(float3(0), 1);
        float4 weights = float4(0.1, 0.2, 0.3, 0.4);
        for (int i = 0; i < pConst.lobeCount; ++i) {
            const float2 uv = float2(1.f) + float2(1.f) * float2(i / 2, i % 2);
            u_gmmStatisticsPack0[index + i] = float4(uv, 0.01, 0.01);
            u_gmmStatisticsPack1[index + i] = float4(0, 1.f, 0, weights[0]);
        }

        return;
    }
    
    // Otherwise, let's do some shading!
    float3 radiance = float3(0);
    PrimaryPayload payload;

    float extra_half = pConst.extra_half_spp ? 2 : 1;

    // Load the GMM statistics
    // --------------------------------------------------------------------
    GMM2D GMMs;
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
            for (int i = 0; i < pConst.lobeCount; ++i) {
                GMMs.sufficientStats0[i] = u_gmmStatisticsPack0Prev[index + i];
                GMMs.sufficientStats1[i] = u_gmmStatisticsPack1Prev[index + i];
            }
        } else {
            float4 weights = float4(0.1, 0.2, 0.3, 0.4);
            for (int i = 0; i < pConst.lobeCount; ++i) {
                const float2 uv = float2(1.f) + float2(1.f) * float2(i / 2, i % 2);
                GMMs.sufficientStats0[i] = float4(uv, 0.01, 0.01);
                GMMs.sufficientStats1[i] = float4(0, 1.f, 0, weights[0]);
            }
        }
    }

    GMMs.build();

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
        const float guided_pdf = GMMs.Pdf(square_coord) / float(2 * k_pi);
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
    const float2 GMMsample = GMMs.DrawSample(float3(GetNextRandomFloat2(RNG), GetNextRandom(RNG)));
    // const float3 guidedDir = CylindricalToUnitVector(UnitToCylindrical(GMMsample));
    const float3 guidedDir = to_world(frame, ConcentricDiskToUniformHemisphere(ToConcentricMap(GMMsample)));
    const float guidedPdf = GMMs.Pdf(GMMsample) / float(2 * k_pi);

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
        const float2 GMMsample = GMMs.DrawSample(float3(GetNextRandomFloat2(RNG), GetNextRandom(RNG)));
        const float3 guidedDir = to_world(frame, ConcentricDiskToUniformHemisphere(ToConcentricMap(GMMsample)));
        const float guidedPdf = GMMs.Pdf(GMMsample) / float(2 * k_pi);

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
    
    for (int i = 0; i < pConst.lobeCount; ++i) {
        u_gmmStatisticsPack0[index + i] = GMMs.sufficientStats0[i];
        u_gmmStatisticsPack1[index + i] = GMMs.sufficientStats1[i];
    }

    if (all(surface.specularF0 == float3(1, 0, 1))) {
        radiance += float3(50.f);
    }

    // radiance += material.emissiveColor;
    const PolymorphicLightInfo light = lights[0];
    radiance += EvaluateDirectLight(primaryRay, surface, light, RNG);
    
    u_color[pixel] = float4(radiance, 1);
}