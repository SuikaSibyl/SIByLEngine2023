#include "../../../include/common/math.hlsli"
#include "../../vxgi/include/vxgi_interface.hlsli"
#include "../include/vxguiding_interface.hlsli"

const StructuredBuffer<uint2> u_irradiance;
const StructuredBuffer<uint4> u_boundMin;
const StructuredBuffer<uint4> u_boundMax;

RWStructuredBuffer<int> u_inverseIndex;
RWStructuredBuffer<uint> u_compactIndex;
RWStructuredBuffer<float> u_PremulIrradiance;

RWStructuredBuffer<uint> u_vplCounter;

const StructuredBuffer<uint> u_representVPLid;
const Texture2D<float4> u_vpl_position;
RWStructuredBuffer<float4> u_representVPL;

[shader("compute")]
[numthreads(512, 1, 1)]
void ComputeMain(uint3 DTid: SV_DispatchThreadID) {
    const uint gid = DTid.x;

    // Store the VPL
    const uint2 irradiance2 = u_irradiance[gid];
    const float irradiance = UnpackIrradiance(irradiance2.x) / irradiance2.y;
    if (irradiance > 0.0f) {
        uint VPLid;
        InterlockedAdd(u_vplCounter[0], 1, VPLid);
        if (VPLid < VXGuider_MAX_CAPACITY) {
            u_compactIndex[VPLid] = gid;
            u_inverseIndex[gid] = VPLid;
            
            uint representVPLid = u_representVPLid[gid];
            int2 pixel_id = int2(representVPLid >> 16, representVPLid & 0xffffu);
            const float4 vpl_position = u_vpl_position[pixel_id];
            if (!all(vpl_position == float4(0))) {
                u_representVPL[VPLid] = vpl_position;
            }

            // Otherwise we need to do some per-voxel work
            AABB aabb; aabb.min = float3(0); aabb.max = float3(1);
            const AABB compact_bound = UnpackCompactAABB(aabb, u_boundMin[gid].xyz, u_boundMax[gid].xyz);
            const float3 extend = compact_bound.max - compact_bound.min;
            float area = 0.f;
            if (extend.x <= extend.y && extend.x <= extend.z) { area = extend.y * extend.z; }
            else if (extend.y <= extend.x && extend.y <= extend.z) { area = extend.x * extend.z; }
            else { area = extend.x * extend.y; }
            u_PremulIrradiance[VPLid] = irradiance * area;
        }
    }
    else {
        u_inverseIndex[gid] = -1;
    }
}