#include "cbt.hlsli"

[[vk::push_constant]]
cbuffer PushConstants {
    uint maxDepth;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    const uint cnt = (1u << maxDepth);
    const uint threadID = dtid.x << 5;
    
    if (threadID < cnt) {
        uint nodeID = threadID + cnt;
        uint alignedBitOffset = cbt__NodeBitID(cbt_CreateNode(nodeID, maxDepth));
        uint bitField = cbt_heap[alignedBitOffset >> 5u];
        uint bitData = 0u;
        
        // 2-bits
        bitField = (bitField & 0x55555555u) + ((bitField >> 1u) & 0x55555555u);
        bitData = bitField;
        cbt_heap[(alignedBitOffset - cnt) >> 5u] = bitData;
        
        // 3-bits
        bitField = (bitField & 0x33333333u) + ((bitField >> 2u) & 0x33333333u);
        bitData = ((bitField >> 0u) & (7u << 0u))
                | ((bitField >> 1u) & (7u << 3u))
                | ((bitField >> 2u) & (7u << 6u))
                | ((bitField >> 3u) & (7u << 9u))
                | ((bitField >> 4u) & (7u << 12u))
                | ((bitField >> 5u) & (7u << 15u))
                | ((bitField >> 6u) & (7u << 18u))
                | ((bitField >> 7u) & (7u << 21u));
        cbt__HeapWriteExplicit(cbt_CreateNode(nodeID >> 2u, maxDepth - 2), 24, bitData);

        // 4-bits
        bitField = (bitField & 0x0F0F0F0Fu) + ((bitField >> 4u) & 0x0F0F0F0Fu);
        bitData = ((bitField >> 0u) & (15u << 0u))
                | ((bitField >> 4u) & (15u << 4u))
                | ((bitField >> 8u) & (15u << 8u))
                | ((bitField >> 12u) & (15u << 12u));
        cbt__HeapWriteExplicit(cbt_CreateNode(nodeID >> 3u, maxDepth - 3), 16, bitData);

        // 5-bits
        bitField = (bitField & 0x00FF00FFu) + ((bitField >> 8u) & 0x00FF00FFu);
        bitData = ((bitField >> 0u) & (31u << 0u))
                | ((bitField >> 11u) & (31u << 5u));
        cbt__HeapWriteExplicit(cbt_CreateNode(nodeID >> 4u, maxDepth - 4), 10, bitData);

        // 6-bits
        bitField = (bitField & 0x0000FFFFu) + ((bitField >> 16u) & 0x0000FFFFu);
        bitData = bitField;
        cbt__HeapWriteExplicit(cbt_CreateNode(nodeID >> 5u, maxDepth - 5), 6, bitData);
    }
}