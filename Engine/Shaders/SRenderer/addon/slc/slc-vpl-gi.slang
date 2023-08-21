#include "../../raytracer/spt_interface.hlsli"
#include "../../raytracer/lights/directional.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../vpl/vpl_interface.hlsli"
#include "../vbuffer/vbuffer.hlsli"
#include "include/slc_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
        uint leaf_start_index;
        float4 bounding_sphere;
        float inv_vpl_paths;
        uint slc_config;
    } pConst;
};

Texture2D<uint4> u_vBuffer;
RWTexture2D<float4> u_color;
RWTexture2D<float4> u_debug;
StructuredBuffer<SLCNode> u_nodeBuffer;

StructuredBuffer<float4> u_vplPositions;
StructuredBuffer<float4> u_vplNormals;
StructuredBuffer<float4> u_vplColors;

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

    const SLCEvaluateConfig config = UnpackSLCEvaluateConfig(pConst.slc_config);
    const VBufferData vhit = UnpackVBufferData(u_vBuffer[pixel]);
    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    if (IsInvalid(vhit)) {
        u_color[pixel] = float4(radiance, 1);
    }
    else {
        const GeometryHit hit = GetGeometryHit(vhit, primaryRay);
        GeometryInfo geometry = geometries[hit.geometryID];
        MaterialInfo material = materials[geometry.materialID];
        float4 albedo = textures[material.baseOrDiffuseTextureIndex].Sample(hit.texcoord, 0);
        

        // shading point infomation
        const float3 p = hit.position;
        const float3 N = hit.geometryNormal;
        const float3 v = normalize(globalUniform.cameraData.posW - p);
        double nprob;
        int nodeChosen = TraverseLightTree(
            1, pConst.leaf_start_index, p, N, v,
            GetNextRandom(RNG), nprob, config, u_nodeBuffer);
            
        SLCNode node = u_nodeBuffer[nodeChosen];
        float4 debugColor = float4(0);

        if (nodeChosen != -1 && node.intensity > 0.0f) {
            const int vplChosen = node.ID;
            const float3 vplPos = u_vplPositions[vplChosen].xyz;
            const float3 vplNormal = u_vplNormals[vplChosen].xyz;
            const float3 vplColor = u_vplColors[vplChosen].xyz;
            float4 ray_direction;
            const float3 atten = AttenFuncVPL(p, N, vplPos, vplNormal, pConst.inv_vpl_paths, ray_direction);

            float3 first_bsdf = EvalBsdf(hit, -primaryRay.direction, ray_direction.xyz);
            first_bsdf /= saturate(dot(N, ray_direction.xyz));
            
            // Trace shadow ray
            Ray shadowRay = SpawnOcclusionRay(hit, vplPos, vplNormal, ray_direction.xyz);
            debugColor = float4(shadowRay.direction, 0);
            shadowRay.tMax = shadowRay.tMax - 0.01f;
            // If not occluded, add the contribution
            if (!TraceOccludeRay(shadowRay, RNG, SceneBVH)) {
                radiance += float3(first_bsdf * atten * vplColor * (nprob == 0.0 ? 0.0 : 1.0 / nprob));
            }
            else {
                debugColor.w = 1.f;
            }
        }

        u_debug[pixel] = debugColor;
        u_color[pixel] = float4(radiance, 1);
    }
}