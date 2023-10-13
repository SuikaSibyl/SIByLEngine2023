#include "../../raytracer/spt_interface.hlsli"
#include "../vbuffer/vbuffer.hlsli"
#include "../vpl/vpl_interface.hlsli"
#include "sst-sampling.hlsli"
#include "../../include/common/light_impl.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
        float inv_vpl_paths;
        int spp;
    } pConst;
};

Texture2D<uint4> u_vBuffer;
RWTexture2D<float4> u_color;
RWTexture2D<float4> u_debug;

const StructuredBuffer<VPLData> u_VPLData;
const ByteAddressBuffer u_vplCounter;

[shader("raygeneration")]
void RgenMain() {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(pConst.resolution);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) {
        return;
    }
    RandomSamplerState RNG = InitRandomSampler(pixel, pConst.sample_batch);
    const Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );

    const int numVPLs = int(u_vplCounter.Load(0));

    const VBufferData vhit = UnpackVBufferData(u_vBuffer[pixel]);
    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    if (IsInvalid(vhit)) {
        u_color[pixel] = float4(radiance, 1);
    }
    else {
        const GeometryHit hit = GetGeometryHit(vhit, primaryRay);
        // shading point infomation
        const float3 p = hit.position;
        const float3 N = hit.geometryNormal;
        const float3 v = normalize(globalUniform.cameraData.posW - p);
        
        for (int i = 0; i < pConst.spp; ++i) {
            float nprob;
            int nodeChosen = TraverseLightTree(
                numVPLs, p, N, v,
                RNG, nprob, u_VPLData);
            // int nodeChosen = clamp(int(GetNextRandom(RNG) * numVPLs), 0, numVPLs - 1);
            // nprob = 1.f / numVPLs;

            if (nprob > 0.f && nodeChosen != -1) {
                const VPLData vpl = u_VPLData[nodeChosen];
                
                // Get position on plane and sample
                const float3 samplePosW = normalPointOnPlane(
                    vpl.getNormW(), vpl.getPosW(), vpl.getVariance(),
                    vpl.getAABBMin(), vpl.getAABBMax(), RNG);

                // u_debug[pixel] = float4(vpl.getPosW(), samplePosW, vpl.getColor());

                const float3 vplPos = samplePosW;
                const float3 vplNormal = vpl.getNormW();
                const float3 vplColor = vpl.getColor();
                float4 ray_direction;
                const float3 atten = AttenFuncVPL(p, N, vplPos, vplNormal, pConst.inv_vpl_paths, ray_direction);
                
                float3 first_bsdf = EvalBsdf(hit, -primaryRay.direction, ray_direction.xyz);
                first_bsdf /= saturate(dot(N, ray_direction.xyz));
                
                // Trace shadow ray
                Ray shadowRay = SpawnOcclusionRay(hit, vplPos, vplNormal, ray_direction.xyz);
                shadowRay.tMax = shadowRay.tMax - 0.01f;
                // If not occluded, add the contribution
                if (!TraceOccludeRay(shadowRay, RNG, SceneBVH)) {
                    radiance += float3(first_bsdf * atten * vplColor * (nprob == 0.0 ? 0.0 : 1.0 / nprob));
                }
            }
        }

        if (any(isnan(radiance))) {
            radiance = float3(0);
        }

        radiance / pConst.spp;
        // radiance += material.emissiveColor;
        const PolymorphicLightInfo light = lights[0];
        radiance += EvaluateDirectLight(primaryRay, hit, light, RNG);

        u_color[pixel] = float4(radiance, 1);
    }
}