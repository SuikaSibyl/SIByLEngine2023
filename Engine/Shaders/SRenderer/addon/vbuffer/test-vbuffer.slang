#include "../../raytracer/spt_interface.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "vbuffer.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
    } pConst;
};

Texture2D<uint4> u_vBuffer;
RWTexture2D<float4> u_color;

[shader("raygeneration")]
void RgenMain() {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(pConst.resolution);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    // If the pixel is outside of the image, don't do anything:
    if(any(pixel >= resolution)) {
        return;
    }
    RandomSamplerState RNG = InitRandomSampler(pixel, pConst.sample_batch);
    Ray primaryRay = generateRay(
        pixel,
        resolution,
        false,
        globalUniform.cameraData
    );

    const VBufferData vhit = UnpackVBufferData(u_vBuffer[pixel]);
    float3 color = float3(0.0f, 0.0f, 0.0f);
    if (IsInvalid(vhit)) {
        u_color[pixel] = float4(color, 1);
    }
    else {
        float3 radiance = float3(0, 0, 0);
        float3 throughput = float3(1, 1, 1);
        float3 di = float3(0, 0, 0);
        float3 gi = float3(0, 0, 0);

        const GeometryHit hit = GetGeometryHit(vhit, primaryRay);
        GeometryInfo geometry = geometries[hit.geometryID];
        MaterialInfo material = materials[geometry.materialID];
        float4 albedo = textures[material.baseOrDiffuseTextureIndex].Sample(hit.texcoord, 0);

        const PolymorphicLightInfo light = lights[0];
        float3 lightColor = unpackLightColor(light);
        float3 lightDir = Unorm32OctahedronToUnitVector(light.databyte0);

        Ray shadowRay = SpawnRay(hit, -lightDir);
        bool occluded = TraceOccludeRay(shadowRay, RNG, SceneBVH);
        float visibility = occluded ? 0.0f : 1.0f;

        di += albedo.xyz * lightColor * saturate(dot(hit.geometryNormal, -lightDir)) * visibility;
        throughput *= albedo.xyz;
        
        u_color[pixel] = float4(di, 1);
    }
}