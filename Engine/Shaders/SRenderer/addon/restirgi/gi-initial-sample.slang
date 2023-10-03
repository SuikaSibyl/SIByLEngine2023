#include "../../include/common/light_impl.hlsli"
#include "../../include/common/packing.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../gbuffer/gbuffer_common.hlsli"
#include "../gbuffer/gbuffer_interface.hlsli"
#include "include/GIReservoir.hlsli"
#include "include/GIResamplingFunctions.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    GIResamplingRuntimeParameters runtimeParams;
    uint initialOutputBufferIndex;
    uint sample_batch;
    bool extra_bounce;
};

RWStructuredBuffer<PackedGIReservoir> u_GIReservoirs;

[shader("raygeneration")]
void RgenMain() {
    // Initialize stage
    // --------------------------------------------------------------------
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = getViewportSize(globalUniform.cameraData);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    RandomSamplerState RNG = InitRandomSampler(pixel, sample_batch);
    
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

    float4 debugval = float4(0,0,0, 1);
    
    GIReservoir reservoir = EmptyGIReservoir();
    float selectedTargetPdf = 0.0f;
    
    if (IsShadingSurfaceValid(surface)) {
        const PolymorphicLightInfo light = lights[0];
        for (int i = 0; i < 1; ++i) {
            float bsdf_pdf;
            Ray bsdfRay = SpawnBsdfRay(surface, -primaryRay.direction, RNG, bsdf_pdf);
            // float3 first_bsdf = EvalBsdf(surface, -primaryRay.direction, bsdfRay.direction);
            // float3 demodulate = sign(surface.diffuseAlbedo) ? float3(1) / surface.diffuseAlbedo : float3(0);
            // float3 throughput = first_bsdf * demodulate;
            float3 throughput = float3(1);
            if (bsdf_pdf == 0) throughput = float3(0, 0, 0);
            const uint32_t RNGState = RNG.state;
            
            PrimaryPayload payload;
            Intersection(bsdfRay, SceneBVH, payload, RNG);
            if (HasHit(payload.hit)) {
                const PolymorphicLightInfo light = lights[0];
                float3 radiance = EvaluateDirectLight(bsdfRay, payload.hit, light, RNG) * throughput;
                if (extra_bounce) {
                    radiance += EvaluateMultibounceIndirect(bsdfRay, 1, throughput, payload, RNG);
                }
                // TODO :: For delta reflection rays, just output the shading result in this shader
                // and don't include it into ReSTIR GI reservoirs.
                // outputShadingResult = isSpecularRay && isDeltaSurface;
                // This pixel has a valid indirect sample so it stores information as an initial GI reservoir.
                GIReservoir local = MakeGIReservoir(payload.hit.position, payload.hit.geometryNormal, radiance, bsdf_pdf, RNGState);
                const float target_pdf = GetGISampleTargetPdfForSurface(local.position, local.radiance, surface, globalUniform.cameraData.posW);
                debugval = float4(radiance, 1);
                if (CombineGIReservoirs(reservoir, local, GetNextRandom(RNG), target_pdf)) {
                    selectedTargetPdf = target_pdf;
                }
                {
                    float3 vec = surface.worldPos - local.position;
                    float distanceToSurface = length(vec);
                    float doted = dot(local.normal, vec / distanceToSurface);
                    float cosine = saturate(doted);
                }
            }
            else {
                reservoir.M++;
            }
        }
    }

    const float normalizationNumerator = 1.0;
    const float normalizationDenominator = selectedTargetPdf * reservoir.M;
    FinalizeGIResampling(reservoir, normalizationNumerator, normalizationDenominator);

    StoreGIReservoir(reservoir, runtimeParams, pixel, initialOutputBufferIndex, u_GIReservoirs);
}