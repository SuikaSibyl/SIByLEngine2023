#include "../../include/common/space_filling_curve.hlsli"
#include "../../include/common/math.hlsli"
#include "sst-interface.hlsli"

const StructuredBuffer<float4> u_vplPositions;
const StructuredBuffer<float4> u_vplNormals;
const StructuredBuffer<float4> u_vplColors;
const ByteAddressBuffer u_vplCounter;

RWStructuredBuffer<uint64_t> u_Codes;
RWStructuredBuffer<VPLData> u_VPLData;
RWStructuredBuffer<TreeConstrIndirectArgs> u_ConstrIndirectArgs;

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    struct PushConstants {
        float3 minExtent;
        float padding0;
        float3 maxExtent;
        float padding1;
    } pConst;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(uint3 tid: SV_DispatchThreadID) {
    const int thread_id = int(tid.x);
    const int numVPLs = int(u_vplCounter.Load(0));
    if (thread_id >= numVPLs) return;
    
    // Compute Morton code
    const float3 position = u_vplPositions[thread_id].xyz;
    const float3 normal = u_vplNormals[thread_id].xyz;
    const float3 radiance = u_vplColors[thread_id].xyz;
    const float3 unipos = (position - pConst.minExtent) / (pConst.maxExtent - pConst.minExtent);
    const uint64_t posCode = ZCurve3DToMortonCode(unipos);
    const uint64_t dirCode = direction_code(normal, 3);
    const uint64_t idCode = thread_id;
    u_Codes[thread_id] = (posCode << 34) | (dirCode << 25) | (idCode << 0);

    // Change the vpl to VPLData,
    // which is a SST-specific data structure
    {   VPLData vpl;
        vpl.setPosW(position);
        vpl.setEarlyStop(0.f);
        vpl.setNormW(normal);
        vpl.setColor(radiance);
        const float intensity = luminance(radiance);
        vpl.setIntensity(intensity);
        vpl.setAABBMin(position);
        vpl.setAABBMax(position);
        vpl.setVariance(float3(0.f));
        vpl.id = thread_id;
        vpl.idChild1 = -1;
        vpl.idChild2 = -1;
        vpl.numVPLSubTree = 0;
        // Write data to buffers
        u_VPLData[thread_id] = vpl;
    }

    // Compute indirect args
    if (thread_id == 0) {
        int numInternalNodes = numVPLs - 1;
        int numTotalNodes = numVPLs * 2 - 1;
        u_ConstrIndirectArgs[0].dispatch_leaf = int3((numVPLs + 255) / 256, 1, 1);
        u_ConstrIndirectArgs[0].dispatch_internal = int3((numInternalNodes + 255) / 256, 1, 1);
        u_ConstrIndirectArgs[0].dispatch_node = int3((numTotalNodes + 255) / 256, 1, 1);
        u_ConstrIndirectArgs[0].numValidVPLs = numVPLs;
        u_ConstrIndirectArgs[0].draw_rects = int4(6, numTotalNodes, 0, 0);
    }
}