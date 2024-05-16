#include "cbt.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    uint passID;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    const uint cnt = (1u << passID);
    const uint threadID = dtid.x;
    if (threadID < cnt) {
        uint nodeID = threadID + cnt;
        uint x0 = cbt_HeapRead(cbt_CreateNode(nodeID << 1u, passID + 1));
        uint x1 = cbt_HeapRead(cbt_CreateNode(nodeID << 1u | 1u, passID + 1));
        cbt__HeapWrite(cbt_CreateNode(nodeID, passID), x0 + x1);
    }
}