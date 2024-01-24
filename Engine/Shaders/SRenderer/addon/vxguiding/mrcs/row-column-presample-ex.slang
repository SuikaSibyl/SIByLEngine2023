#include "../../vbuffer/vbuffer.hlsli"
#include "../../vxgi/include/conetrace_utils.hlsli"
#include "../../vxgi/include/vxgi_interface.hlsli"
#include "../geometry/common.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    // int2 map_size;
    int2 resolution;
    int rand_seed;
};

const Texture2D<float4> u_vpl_position;
const Texture2D<uint4> u_vBuffer;
const StructuredBuffer<int> u_inverse_index;
const StructuredBuffer<uint> u_vplCounter;
const StructuredBuffer<uint> u_hashRecord;

RWStructuredBuffer<float4> u_RepresentPixel;
RWStructuredBuffer<float4> u_RepresentVPL;
RWStructuredBuffer<uint4> u_IndirectArgs;

#define VPL_BOUND_ENABLE
#ifdef VPL_BOUND_ENABLE
RWStructuredBuffer<uint> u_BoundMin;
RWStructuredBuffer<uint> u_BoundMax;
#endif // VPL_BOUND_ENABLE
// RWStructuredBuffer<uint> u_DirectionHash;

uint PackInt2ToUInt(int2 xy) {
    return (uint(xy.x) << 16) | uint(xy.y);
}

void InjectVPLInfo(
    float3 vpl_position,
    float3 vpl_direction,
    int3 voxelID,
    int compactID
) {
    // Inject the position
    const VoxelTexInfo info = GetVoxelTexInfo();
    const AABB voxel_bound = VoxelToBound(voxelID, 0, info);
    const int voxelIDFlatten = FlatIndex(voxelID, info.volumeDimension);
    uint max_uint = 0xffffffff;
    const float3 voxel_bound_extend = voxel_bound.max - voxel_bound.min;
    uint3 pRelative = uint3(saturate((vpl_position - voxel_bound.min) / voxel_bound_extend) * max_uint);
    {   // Inject the bound
        InterlockedMin(u_BoundMin[compactID * 4 + 0], pRelative.x);
        InterlockedMin(u_BoundMin[compactID * 4 + 1], pRelative.y);
        InterlockedMin(u_BoundMin[compactID * 4 + 2], pRelative.z);
        InterlockedMax(u_BoundMax[compactID * 4 + 0], pRelative.x);
        InterlockedMax(u_BoundMax[compactID * 4 + 1], pRelative.y);
        InterlockedMax(u_BoundMax[compactID * 4 + 2], pRelative.z);
        // Inject the count
        InterlockedAdd(u_BoundMin[compactID * 4 + 3], 1);
        u_BoundMax[compactID * 4 + 3] = voxelIDFlatten;
    }
    // // Inject the direction
    // const int directionHash = hash_directional(vpl_direction);
    // InterlockedOr(u_DirectionHash[compactID * 2 + (directionHash / 32)],
    //               (1u << (directionHash % 32)));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(
    int3 DTid: SV_DispatchThreadID,
    int3 Gid: SV_GroupID
) {
    const int2 pixelID = DTid.xy;
    const Ray primaryRay = generateRay(
        pixelID,
        resolution,
        false,
        globalUniform.cameraData
    );
    
    // Grab the bsdf sampled buffer
    const VoxelTexInfo info = GetVoxelTexInfo();
    const float4 vpl_position = u_vpl_position[pixelID];
    if (!all(vpl_position == float4(0))) {
        const int3 vxID = int3(WorldToVoxel(vpl_position.xyz, info) * info.volumeDimension);
        uint32_t bucketID = QueryBucketID(vxID, u_config);
        uint32_t key = QueryKey(vxID);
        int cellID = FindCellID(bucketID, key, u_config, u_hashRecord);
        int globalID = GetGlobalID(bucketID, cellID, u_config);
        const int compactID = u_inverse_index[globalID];
        // Inject the voxel
        u_RepresentVPL[compactID] = vpl_position;
        InjectVPLInfo(vpl_position.xyz, Unorm32OctahedronToUnitVector(asuint(vpl_position.w)), vxID, compactID);
    }

    // Choosing 128 pixels for row sampling
    if (pixelID.x < 16 && pixelID.y < 8) {
        // stratified sampling from 16 x 8 grid
        RandomSamplerState RNG = InitRandomSampler(pixelID, rand_seed);
        const int2 cell_id = pixelID;
        const int flatten_id = pixelID.y * 16 + pixelID.x;
        const float2 cell_size = float2(resolution) / float2(16, 8);
        const float2 pixel = cell_size * (float2(cell_id) + GetNextRandomFloat2(RNG));
        const int2 pixel_int = clamp(int2(pixel), int2(0, 0), resolution - int2(1, 1));
        
        const VBufferData vhit = UnpackVBufferData(u_vBuffer[pixel_int]);
        if (!IsInvalid(vhit)) {
            const GeometryHit hit = GetGeometryHit(vhit, primaryRay);
            u_RepresentPixel[flatten_id] = float4(hit.position, UnitVectorToUnorm32Octahedron(hit.shadingNormal));
        } else {
            u_RepresentPixel[flatten_id] = float4(0, 0, 0, 0);
        }
    }
    
    if (all(pixelID == int2(0, 0))) {
        const uint vplCount = u_vplCounter[0];
        u_IndirectArgs[0] = uint4((vplCount + 255) / 256, 1, 1, vplCount);
        u_IndirectArgs[1] = int4((vplCount * 2 - 1 + 255) / 256, 1, 1, vplCount);
        u_IndirectArgs[2] = uint4(4, (vplCount + 7) / 8, 1, vplCount);
    }
}