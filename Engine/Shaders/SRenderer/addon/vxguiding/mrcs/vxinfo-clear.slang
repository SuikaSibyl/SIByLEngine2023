#include "../../vxgi/include/vxgi_interface.hlsli"
#include "../include/vxguiding_interface.hlsli"

#define VPL_BOUND_ENABLE
#ifdef VPL_BOUND_ENABLE
RWStructuredBuffer<uint4> u_BoundMin;
RWStructuredBuffer<uint4> u_BoundMax;
#endif // VPL_BOUND_ENABLE
// RWStructuredBuffer<uint2> u_DirectionHash;
RWStructuredBuffer<float> u_PremulIrradiance;

cbuffer VoxelizerUniform { VoxerlizerData gVD; }
const ByteAddressBuffer u_vxCounter;
const StructuredBuffer<uint> u_CompactIndex;
const Texture3D<uint> u_IrradianceMap;
const Texture3D<uint> u_VPLCount;
const StructuredBuffer<uint4> u_pMin;
const StructuredBuffer<uint4> u_pMax;

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(int3 DTid: SV_DispatchThreadID) {
    const uint compactID = DTid.x;
    const int maxID = int(u_vxCounter.Load(0));
    u_BoundMin[compactID] = uint4(uint3(0xffffffff), 0);
    u_BoundMax[compactID] = uint4(uint3(0), 0);
    // u_DirectionHash[compactID] = uint2(0, 0);
    
    if (compactID >= maxID) { return; }

    // Otherwise we need to do some per-voxel work
    const uint voxelID = u_CompactIndex[compactID];
    const int3 mapPos = ReconstructIndex(voxelID, 64);
    // Compute the irradiance * surface area
    const float irradiance = UnpackIrradiance(u_IrradianceMap[mapPos]) / u_VPLCount[mapPos];
    const VoxelTexInfo info = CreateVoxelTexInfo(gVD);
    const AABB aabb = VoxelToBound(mapPos, 0, info);
    const AABB compact_bound = UnpackCompactAABB(aabb, u_pMin[voxelID].xyz, u_pMax[voxelID].xyz);
    const float3 extend = compact_bound.max - compact_bound.min;
    float area = 0.f;
    if (extend.x <= extend.y && extend.x <= extend.z) { area = extend.y * extend.z; }
    else if (extend.y <= extend.x && extend.y <= extend.z) { area = extend.x * extend.z; }
    else { area = extend.x * extend.y; }
    u_PremulIrradiance[DTid.x] = irradiance * area;
}