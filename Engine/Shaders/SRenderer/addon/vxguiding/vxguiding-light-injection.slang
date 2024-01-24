#include "../../include/common/light_impl.hlsli"
#include "../../include/common/random.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/raytracer_descriptor_set.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../vbuffer/vbuffer.hlsli"
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

RWTexture3D<uint> u_pIrradiance;
RWTexture3D<uint> u_vplCount;
RWTexture2D<float4> u_shadingpoint;

const Texture2D<float4> u_position;
const Texture2D<uint4> u_vBuffer;
const Texture2D<float> u_intensity;

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
    const uint packedIrr = PackIrradiance(irradiance, 100);
    if (use_avg_not_max) {
        InterlockedAdd(u_pIrradiance[voxelPos], packedIrr);
        InterlockedAdd(u_vplCount[voxelPos], 1);
    } else {
        InterlockedMax(u_pIrradiance[voxelPos], packedIrr);
        u_vplCount[voxelPos] = 1;
    }
}

[shader("compute")]
[numthreads(16, 16, 1)] // 16x16x1 threads in a group
void ComputeMain(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    uint gi: SV_GroupIndex
) {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(pConst.resolution);
    const int2 pixel = int2(dtid.xy);
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
        u_shadingpoint[pixel] = float4(k_inf, k_inf, k_inf, 1);
    }
    else {
        const GeometryHit hit = GetGeometryHit(vhit, primaryRay);
        u_shadingpoint[pixel] = float4(hit.position, asfloat(UnitVectorToUnorm32Octahedron(hit.geometryNormal)));
    }

    const float3 vpl_position = u_position[pixel].xyz;
    const float irradiance = u_intensity[pixel];
    
    if (irradiance <= 0.f) return;

    // inject irradiance to voxel where the hit position is
    // const bool inject_1st_vertex = (pConst.setting & 0b1) == 0b1;
    const bool use_avg_not_max = (pConst.setting & 0b10) == 0b10;
    VoxelTexInfo info = CreateVoxelTexInfo(gVD);
    InjectVPL(vpl_position, irradiance, info, use_avg_not_max);
}