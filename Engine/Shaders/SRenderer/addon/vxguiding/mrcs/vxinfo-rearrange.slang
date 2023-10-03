// RWStructuredBuffer<uint4> u_BoundMinOrdered;
// RWStructuredBuffer<uint4> u_BoundMaxOrdered;
// RWStructuredBuffer<uint2> u_DirectionHashOrdered;

const StructuredBuffer<uint4> u_BoundMin;
const StructuredBuffer<uint4> u_BoundMax;
// const StructuredBuffer<uint2> u_DirectionHash;

const StructuredBuffer<int4> u_IndirectArgs;
const StructuredBuffer<int>  u_compact2leaf;

RWStructuredBuffer<uint4> u_pMin;
RWStructuredBuffer<uint4> u_pMax;

[[vk::push_constant]]
cbuffer PushConstants {
    int threshold_use_vpl_bound;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(uint3 dtid: SV_DispatchThreadID) {
    const uint gid = dtid.x;
    const uint vxcount = u_IndirectArgs[0].w;
    const int numInternalNodes = int(vxcount) - 1;
    if (gid >= vxcount) { return; }

    const uint4 bound_min = u_BoundMin[gid];
    const uint4 bound_max = u_BoundMax[gid];
    // const uint2 direction_hash = u_DirectionHash[gid];
    
    // // write to ordered buffer
    // const uint leaf_id = u_compact2leaf[gid] - numInternalNodes;
    // u_BoundMinOrdered[leaf_id] = bound_min;
    // u_BoundMaxOrdered[leaf_id] = bound_max;
    // u_DirectionHashOrdered[leaf_id] = direction_hash;

    // write to pMin and pMax if the number of vpl is large enough
    if (threshold_use_vpl_bound >= 0 &&
        bound_min.w > threshold_use_vpl_bound) {
        const int voxelIDFlatten = bound_max.w;
        u_pMin[voxelIDFlatten] = bound_min;
        u_pMax[voxelIDFlatten] = bound_max;
    }
}