#include "../../include/common/math.hlsli"
#include "../vxgi/include/vxgi_interface.hlsli"
#include "include/vxguiding_interface.hlsli"

RWTexture3D<uint> u_pIrradiance;
RWTexture3D<int> u_pInverseIndex;
RWStructuredBuffer<uint> u_pCompactIndex;

RWStructuredBuffer<uint> u_vplCounter;

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}
[shader("compute")]
[numthreads(8, 8, 8)] // 8x8x8 threads in a group
void ComputeMain(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID, uint gi: SV_GroupIndex) {
    const uint3 voxelToFetch = gid * 8 + gtid;
    int voxelIDFlatten = FlatIndex(int3(voxelToFetch), gVD.voxelSize);

    // Store the VPL
    float irradiance = UnpackIrradiance(u_pIrradiance[voxelToFetch]);
    if (irradiance > 0.0f) {
        uint VPLid;
        InterlockedAdd(u_vplCounter[0], 1, VPLid);
        if (VPLid < VXGuider_MAX_CAPACITY) {
            u_pCompactIndex[VPLid] = voxelIDFlatten;
            u_pInverseIndex[voxelToFetch] = VPLid;
        }
    }
    else {
        u_pInverseIndex[voxelToFetch] = -1;
    }
}