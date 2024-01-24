#include "../../../include/common/light_impl.hlsli"
#include "../../../include/common/random.hlsli"
#include "../../../include/common/sampling.hlsli"
#include "../../../include/raytracer_descriptor_set.hlsli"
#include "../../../include/scene_descriptor_set.hlsli"
#include "../../../raytracer/spt_interface.hlsli"
#include "../../vbuffer/vbuffer.hlsli"
#include "../../vxgi/include/conetrace_utils.hlsli"
#include "../../vxgi/include/vxgi_interface.hlsli"
#include "../include/vxguiding_interface.hlsli"
#include "../geometry/common.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    struct PushConstants {
        uint2 resolution;
        uint sample_batch;
    } pConst;
};

Texture2D<uint4> u_vBuffer;
RWStructuredBuffer<uint> u_hashRecord;
RWStructuredBuffer<uint> u_irradiance;
RWStructuredBuffer<uint> u_representVPLid;

RWTexture2D<float4> u_shadingpoint; // shading point with compressed normal
RWTexture2D<float4> u_position;     // first bounce hitpoint with compressed normal
RWTexture2D<float4> u_color;        // color of the 1st bounce global illumination

void InjectVPL(
    in_ref(float3) position,
    in_ref(float) irradiance,
    in_ref(VoxelTexInfo) info,
    in_ref(int2) pixel,
) {
    const int3 voxelID = int3(WorldToVoxel(position, info) * info.volumeDimension);
    const uint32_t bucketID = QueryBucketID(voxelID, u_config);
    const uint32_t fingerprint = QueryKey(voxelID);
    int cellID = QueryCellID(bucketID, fingerprint, u_config, u_hashRecord);
    if (cellID == -1) return;
    
    uint32_t globalID = GetGlobalID(bucketID, cellID, u_config);
    const uint packedIrr = PackIrradiance(irradiance);
    InterlockedAdd(u_irradiance[globalID * 2 + 0], packedIrr);
    InterlockedAdd(u_irradiance[globalID * 2 + 1], 1);
    
    uint32_t vplid = (pixel.x << 16) + pixel.y;
    u_representVPLid[globalID] = vplid;
}

[shader("raygeneration")]
void RgenMain() {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(pConst.resolution);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;
    RandomSamplerState RNG = InitRandomSampler(pixel, pConst.sample_batch);
    const Ray primaryRay = generateRay(pixel, resolution, false, globalUniform.cameraData);

    const VBufferData vhit = UnpackVBufferData(u_vBuffer[pixel]);
    // if the hit is an invalid hit, then simply ignore the thing
    if (IsInvalid(vhit)) {
        u_color[pixel] = float4(0);
        u_position[pixel] = float4(0);
        u_shadingpoint[pixel] = float4(k_inf, k_inf, k_inf, 1);
        return;
    }
    // otherwise unpack the vbuffer hitpoint
    const GeometryHit hit = GetGeometryHit(vhit, primaryRay);
    // and immediately output the first hitpoint with compressed geometry normal
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
    
    VoxelTexInfo info = GetVoxelTexInfo();
    InjectVPL(payload.hit.position, irradiance, info, pixel);
}