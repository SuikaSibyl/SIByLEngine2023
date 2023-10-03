RWStructuredBuffer<uint4> u_pMin;
RWStructuredBuffer<uint4> u_pMax;
RWTexture3D<uint> u_pIrradiance;
RWTexture3D<uint> u_vplCount;

StructuredBuffer<uint4> u_pMinPrebake;
StructuredBuffer<uint4> u_pMaxPrebake;

#include "../../include/common/math.hlsli"
#include "../vxgi/include/vxgi_interface.hlsli"

[shader("compute")]
[numthreads(8, 8, 8)] // 8x8x8 threads in a group
void ComputeMain(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID, uint gi: SV_GroupIndex) {
    const uint3 voxelToFetch = gid * 8 + gtid;
    int voxelIDFlatten = FlatIndex(int3(voxelToFetch), 64);
    
    // u_pMin[voxelIDFlatten] = uint4(uint3(0xffffffff), 0);
    // u_pMax[voxelIDFlatten] = uint4(uint3(0), 0);
    u_pMin[voxelIDFlatten] = u_pMinPrebake[voxelIDFlatten];
    u_pMax[voxelIDFlatten] = u_pMaxPrebake[voxelIDFlatten];
    u_pIrradiance[voxelToFetch] = 0;
    u_vplCount[voxelToFetch] = 0;
}