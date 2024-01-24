#include "../../include/common/light_impl.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../../raytracer/spt_interface.hlsli"
#include "../vbuffer/vbuffer.hlsli"
#include "../vxgi/include/conetrace_utils.hlsli"
#include "../vxgi/include/vxgi_interface.hlsli"
#include "include/vxguiding_interface.hlsli"


[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
        uint setting;
    } pConst;
};

Texture2D<uint4> u_vBuffer;
RWTexture3D<uint> u_pIrradiance;
RWTexture3D<uint> u_vplCount;
RWTexture2D<float4> u_position;
RWTexture2D<float4> u_color;
RWTexture2D<float4> u_shadingpoint;

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

void InjectVPL(
    in_ref(float3) position,
    in_ref(float) irradiance,
    in_ref(VoxelTexInfo) info,
    bool use_avg_not_max
) {
    const int3 voxelPos = int3(WorldToVoxel(position, info) * info.volumeDimension);
    const int voxelIDFlatten = FlatIndex(voxelPos, info.volumeDimension);
    const uint packedIrr = PackIrradiance(irradiance);
    if (use_avg_not_max) {
        InterlockedAdd(u_pIrradiance[voxelPos], packedIrr);
        InterlockedAdd(u_vplCount[voxelPos], 1);
    } else {
        InterlockedMax(u_pIrradiance[voxelPos], packedIrr);
        u_vplCount[voxelPos] = 1;
    }
}

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

    const VBufferData vhit = UnpackVBufferData(u_vBuffer[pixel]);
    if (IsInvalid(vhit)) {
        u_color[pixel] = float4(0);
        u_position[pixel] = float4(0);
        u_shadingpoint[pixel] = float4(k_inf, k_inf, k_inf, 1);
        return;
    }
    
    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);
    u_shadingpoint[pixel] = float4(hit.position, asfloat(UnitVectorToUnorm32Octahedron(hit.geometryNormal)));

    // Sample the next light ray if we need further bounces
    float bsdf_pdf;
    Ray bsdfRay = SpawnBsdfRay(hit, -primaryRay.direction, RNG, bsdf_pdf);
    if (bsdf_pdf == 0) {
        u_color[pixel] = float4(0);
        u_position[pixel] = float4(0);
        return;
    }
    
    PrimaryPayload payload;
    Intersection(bsdfRay, SceneBVH, payload, RNG);
    // If we didn't hit anything, we're done
    if (HasHit(payload.hit) == false) {
        u_color[pixel] = float4(0);
        u_position[pixel] = float4(0);
        return;
    }

    const PolymorphicLightInfo light = lights[0];
    const float3 color = EvaluateDirectLight(bsdfRay, payload.hit, light, RNG);
    const float irradiance = maxComponent(color);
    const uint hitNormal = UnitVectorToUnorm32Octahedron(payload.hit.geometryNormal);
    
    if (irradiance != 0) {
        u_color[pixel] = float4(color, 1);
        u_position[pixel] = float4(payload.hit.position, asfloat(hitNormal));
    }
    else {
        u_color[pixel] = float4(0);
        u_position[pixel] = float4(0);
        return;
    }
    
    // inject irradiance to voxel where the hit position is
    const bool inject_1st_vertex = (pConst.setting & 0b1) == 0b1;
    const bool use_avg_not_max = (pConst.setting & 0b10) == 0b10;

    VoxelTexInfo info = CreateVoxelTexInfo(gVD);
    InjectVPL(payload.hit.position, irradiance, info, use_avg_not_max);
}