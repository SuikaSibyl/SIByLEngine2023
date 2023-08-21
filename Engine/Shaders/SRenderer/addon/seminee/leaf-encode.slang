#include "seminee_interface.hlsli"
#include "../../include/common/space_filling_curve.hlsli"

RWStructuredBuffer<VPLData>     u_VPLs;
RWStructuredBuffer<uint64_t>    u_Codes;

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    struct PushConstants {
        float3 minExtent;
        float3 maxExtent;
        uint   numVPLs;
    } pConst;
};

uint64_t direction_code_crzy(float3 normal) {
    uint64_t code = 0;
    const float3 absn = abs(normal);
    if (absn.x >= absn.y && absn.x >= absn.z) {
        if(normal.x >= 0) code = 0;
        else code = 1;
    }
    else if (absn.y >= absn.x && absn.y >= absn.z) {
        if(normal.y >= 0) code = 2;
        else code = 3;
    }
    else {
        if(normal.z >= 0) code = 4;
        else code = 5;
    }
    return code;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void LeafEncodeMain(uint3 tid: SV_DispatchThreadID) {
    uint thread_id = tid.x;
    if (thread_id >= pConst.numVPLs) return;

    const VPLData vpl = u_VPLs[thread_id];
    if (vpl.isInvalid()) { // VPL is not valid
        // Set code to maximum so it will always end up in the end after sorting.
        u_Codes[thread_id] = uint64_t(-1);
        return;
    }

    const float3 position = u_VPLs[tid.x].getPosW();
    const float3 normal = u_VPLs[tid.x].getNormW();
    const float3 unipos = (position - pConst.minExtent) / (pConst.maxExtent - pConst.minExtent);
    const uint64_t posCode = ZCurve3DToMortonCode(unipos);
    const uint64_t dirCode = direction_code(normal, 3);
    const uint64_t dirCodeCrazy = direction_code_crzy(normal);
    const uint64_t idCode = thread_id;

    u_Codes[thread_id] = (dirCodeCrazy << 61) | (posCode << 31) | (dirCode << 22) | (idCode << 0);
    // u_Codes[thread_id] = (posCode << 34) | (dirCode << 25) | ((idCode) << 0);
}