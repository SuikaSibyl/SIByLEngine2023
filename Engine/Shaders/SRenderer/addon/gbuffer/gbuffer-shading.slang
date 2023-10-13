#include "../../include/common/light_impl.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/common/packing.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "gbuffer_common.hlsli"
#include "gbuffer_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    uint sample_batch;
};

RWTexture2D<uint> u_Diffuse;
RWTexture2D<uint> u_Specular;

RWTexture2D<uint> u_RNGSeed;
RWTexture2D<uint> u_RNGPrev;

RWTexture2D<float4> u_Debug;

RWTexture2D<float4> u_Indicator;

int2 IdxUnpack(float a) {
    uint x = asuint(a);
    return int2((x >> 16), x & 0xFFFF);
}

[shader("raygeneration")]
void RgenMain() {
    // Initialize stage
    // --------------------------------------------------------------------
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = getViewportSize(globalUniform.cameraData);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    RandomSamplerState RNG;
    RNG.state = u_RNGSeed[pixel];
    u_RNGPrev[pixel] = RNG.state;
    
    // Load Primary Hit from GBuffer
    // --------------------------------------------------------------------
    const Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );
    
    ShadingSurface surface = GetGBufferSurface(
        pixel, globalUniform.cameraData
    );

    SplitShading shading;
    shading.diffuse = float3(0);
    shading.specular = float3(0);
    const int spp = 2;
    
    if (IsShadingSurfaceValid(surface)) {
        PrimaryPayload payload;
        float3 throughput = float3(1);
        for (int i = 0; i < spp; ++i) {

            float bsdf_pdf;
            Ray bsdfRay = SpawnBsdfRay(surface, -primaryRay.direction, RNG, bsdf_pdf);
            SplitShading gi = EvalBsdfSplit(surface, -primaryRay.direction, bsdfRay.direction);

            PrimaryPayload payload;
            Intersection(bsdfRay, SceneBVH, payload, RNG);
            if (HasHit(payload.hit)) {
                const PolymorphicLightInfo light = lights[0];
                float3 radiance = EvaluateDirectLight(bsdfRay, payload.hit, light, RNG);
                // further bounces
                radiance += EvaluateMultibounceIndirect(bsdfRay, 1, throughput, payload, RNG);
                shading.diffuse += radiance * gi.diffuse / bsdf_pdf;
                shading.specular += radiance * gi.specular / bsdf_pdf;
            }
        }
        
        // Intersection(bsdfRay, SceneBVH, payload, RNG);
        // if (HasHit(payload.hit)) {
        //     const PolymorphicLightInfo light = lights[0];
        //     const float3 radiance = EvaluateDirectLight(bsdfRay, payload.hit, light, RNG);
        //     shading.diffuse += radiance * gi.diffuse / bsdf_pdf;
        //     shading.specular += radiance * gi.specular / bsdf_pdf;
        // }
    }
    else {
        u_Debug[pixel] = float4(0, 0, 0, 0);
    }

    shading.diffuse /= spp;
    shading.specular /= spp;
    
    if (IsShadingSurfaceValid(surface)) {
        // Add direct light
        const PolymorphicLightInfo light = lights[0];
        SplitShading di = EvaluateDirectLightSplit(primaryRay, surface, light, RNG);
        shading = add(shading, di);
    }

    RNG = InitRandomSampler(pixel, sample_batch);
    u_RNGSeed[pixel] = RNG.state;
    u_Diffuse[pixel] = PackRGBE(shading.diffuse);
    u_Specular[pixel] = PackRGBE(shading.specular);

    // if (all(pixel < resolution / 3)) {
    //     float3 diffuse = float3(0);
        
    //     ShadingSurface surface = GetGBufferSurface(
    //         IdxUnpack(u_Indicator[pixel].x), globalUniform.cameraData
    //     );
    //     ShadingSurface prevsurf = GetPrevGBufferSurface(
    //         IdxUnpack(u_Indicator[pixel].y), globalUniform.cameraData
    //     );
    //     RandomSamplerState iRNG;
    //     iRNG.state = asuint(u_Indicator[pixel].z);
        
    //     if (IsShadingSurfaceValid(prevsurf)) {
    //         const PolymorphicLightInfo light = lights[0];
    //         const float3 di = EvaluateDirectLight(primaryRay, surface, light, iRNG);

    //         float bsdf_pdf;
    //         Ray bsdfRay = SpawnBsdfRay(surface, -primaryRay.direction, iRNG, bsdf_pdf);
    //         float3 first_bsdf = EvalBsdf(surface, -primaryRay.direction, bsdfRay.direction);
    //         float3 demodulate = (sign(surface.diffuseAlbedo) == float3(1)) ? float3(1) / surface.diffuseAlbedo : float3(0);
    //         float3 throughput = first_bsdf / bsdf_pdf * demodulate;
    //         if (bsdf_pdf == 0) throughput = float3(0, 0, 0);

    //         PrimaryPayload payload;
    //         Intersection(bsdfRay, SceneBVH, payload, iRNG);
    //         if (HasHit(payload.hit)) {
    //             const PolymorphicLightInfo light = lights[0];
    //             diffuse += EvaluateDirectLight(bsdfRay, payload.hit, light, iRNG) * throughput;
    //         }

    //         float a = luminance(UnpackRGBE(PackRGBE(diffuse)));
    //         // u_Debug[pixel] = float4(a, u_Indicator[pixel].x, u_Indicator[pixel].y, 1);
    //     }

    //     // if (all(surface.worldPos == prevsurf.worldPos)
    //     //     && all(surface.shadingNormal == prevsurf.shadingNormal)
    //     //     && all(surface.geometryNormal == prevsurf.geometryNormal)) {
    //     //     u_Debug[pixel] = float4(0, 0, 0, 1);
    //     // }
    //     // else {
    //     //     u_Debug[pixel] = float4(1, 0, 1, 1);
    //     // }
    // }
    // else {
    //     // u_Debug[pixel] = float4(0, 0, 0, 1);
    // }
}