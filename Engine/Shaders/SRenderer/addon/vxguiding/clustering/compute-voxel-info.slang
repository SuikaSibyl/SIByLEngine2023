#include "../../../include/common/cpp_compatible.hlsli"
#include "../../vxgi/include/vxgi_interface.hlsli"
#include "../include/vxguiding_interface.hlsli"
#include "voxel_cluster.hlsli"

StructuredBuffer<uint4> u_pMin;
StructuredBuffer<uint4> u_pMax;

ByteAddressBuffer u_vxCounter;
StructuredBuffer<uint> u_pCompactIndex;
const Texture3D<uint> u_IrradianceMap;
const Texture3D<uint> u_VPLCount;

RWStructuredBuffer<float4> u_vxNormal;
RWStructuredBuffer<float> u_vxIrradiance;
RWStructuredBuffer<int> u_associate;

cbuffer VoxelizerUniform {
    VoxerlizerData gVD;
}

/** Load the 27 neighbors of the voxel at the given position. */
void LoadNeighbors(
    inout_ref(uint) neighbors[3][3][3], 
    in_ref(int3) pos
) {
    int3 offset = int3(0);
    for (offset.z = -1; offset.z <= 1; offset.z++) {
        for (offset.y = -1; offset.y <= 1; offset.y++) {
            for (offset.x = -1; offset.x <= 1; offset.x++) {
                int3 neighborPos = pos + offset;
                if (any(neighborPos < 0 || neighborPos >= 64)) {
                    neighbors[offset.x + 1][offset.y + 1][offset.z + 1] = 0;
                }
                else {
                    neighbors[offset.x + 1][offset.y + 1][offset.z + 1] = u_IrradianceMap[neighborPos].x;
                }
            }
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int Gid: SV_GroupIndex
) {
    const int maxID = int(u_vxCounter.Load(0));
    if (DTid.x >= maxID) return;

    const uint voxelID = u_pCompactIndex[DTid.x];

    uint neighbors[3][3][3];
    const int3 mapPos = ReconstructIndex(voxelID, 64);
    LoadNeighbors(neighbors, mapPos);
    const float3 normalColor = GetNormal(neighbors);
    u_vxNormal[DTid.x] = float4(normalize(normalColor), 0.0f);
    u_associate[DTid.x] = -1;

    float irradiance = UnpackIrradiance(u_IrradianceMap[mapPos]) / u_VPLCount[mapPos];
    
    VoxelTexInfo info = CreateVoxelTexInfo(gVD);
    AABB aabb = VoxelToBound(mapPos, 0, info);
    AABB compact_bound = UnpackCompactAABB(aabb, u_pMin[voxelID].xyz, u_pMax[voxelID].xyz);

    float3 extend = compact_bound.max - compact_bound.min;
    float area = 0.f;
    if(extend.x <= extend.y && extend.x <= extend.z) {
        area = extend.y * extend.z;
    }
    else if(extend.y <= extend.x && extend.y <= extend.z) {
        area = extend.x * extend.z;
    }
    else {
        area = extend.x * extend.y;
    }
    u_vxIrradiance[DTid.x] = irradiance * area;
}