RWStructuredBuffer<uint4> u_pMin;
RWStructuredBuffer<uint4> u_pMax;

#include "../../include/common/math.hlsli"
#include "../vxgi/include/vxgi_interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    uint resolution;
};

[shader("compute")]
[numthreads(8, 8, 8)] // 8x8x8 threads in a group
void ComputeMain(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID, uint gi: SV_GroupIndex) {
    const uint3 voxelToFetch = gid * 8 + gtid;
    int voxelIDFlatten = FlatIndex(int3(voxelToFetch), resolution);
    
    u_pMin[voxelIDFlatten] = uint4(uint3(0xffffffff), 0);
    u_pMax[voxelIDFlatten] = uint4(uint3(0), 0);
}