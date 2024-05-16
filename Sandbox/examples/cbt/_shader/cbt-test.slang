#include "cbt/cbt-sbt.hlsli"
#include "cbt/cbt.hlsli"


[[vk::push_constant]]
cbuffer PushConstants {
    float3 center;
    float radius;
    int stop_split;
};

struct DebugStruct {
    float4 data1;
    float4 data2;
};

RWStructuredBuffer<DebugStruct> debug_buffer;

float sdSphere(float3 p, float3 c, float s) {
    return length(p - c) - s;
}

float MinDistanceToAABB(float3 point, float3 boxMin, float3 boxMax) {
    float3 closestPoint = clamp(point, boxMin, boxMax);
    float3 distances = point - closestPoint;
    float minDistanceSquared = dot(distances, distances);
    return sqrt(minDistanceSquared);
}

float MaxDistanceToAABB(float3 point, float3 boxMin, float3 boxMax) {
    float3 extents = (boxMax - boxMin) * 0.5;
    float3 center = (boxMin + boxMax) * 0.5;
    float3 closestPoint = clamp(point, boxMin, boxMax);
    float3 distances = abs(point - center) - extents;
    float maxDistance = max(max(distances.x, distances.y), distances.z);
    return maxDistance;
}

bool should_split(AABB bounds) {
    return MinDistanceToAABB(center, bounds.min, bounds.max) < radius;
    // float3 size = bounds.max - bounds.min;
    // int positive_count = 0;
    // int negative_count = 0;
    // for (int i = 0; i < 8; ++i) {
    //     const float3 p = bounds.min +
    //                      float3((i & 1) * size.x,
    //                             ((i >> 1) & 1) * size.y,
    //                             ((i >> 2) & 1) * size.z);
    //     float dist = sdSphere(p, center, radius);
    //     if (dist > 0) positive_count++;
    //     else negative_count++;
    // }
    // return positive_count > 0 && negative_count > 0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(
    int3 dtid: SV_DispatchThreadID,
    int3 gtid: SV_GroupThreadID,
    int3 gid: SV_GroupID,
    int gi: SV_GroupIndex
) {
    int node_count = cbt_NodeCount();
    if (dtid.x >= node_count) return;

    cbt_Node node = cbt_DecodeNode(dtid.x);
    AABB bounds = nodeID2AABB(node.id, node.depth);
    int max_depth = cbt_MaxDepth();
    // for eight points of the cube

    int do_split = 0;
    int do_merge = 0;

    if (stop_split == 0) {
        if (should_split(bounds)) {
            cbt_SplitNode(node);
            do_split = 1;
        }
        else {
            if (node.id % 2 == 0) {
                cbt_Node rs = cbt_RightSiblingNode(node);
                cbt_Node parent = cbt_ParentNode(node);
                AABB rbounds = nodeID2AABB(rs.id, rs.depth);
                bool rs_should_split = should_split(rbounds);
                AABB pbounds = nodeID2AABB(parent.id, parent.depth);
                if (cbt_IsLeafNode(rs) && cbt_IsLeafNode(node) &&
                    !should_split(pbounds) && !rs_should_split) {
                    cbt_MergeNode(node);
                }
            }
        }
    }
    else if (stop_split == 1) {
        if (node.id % 2 == 0) {
            cbt_MergeNode(node);
        }
    }
    
    if (dtid.x < 32) {
        DebugStruct data;
        data.data1 = float4(bounds.min, do_split);
        data.data2 = float4(bounds.max, do_merge);
        debug_buffer[dtid.x] = data;
    }

    // if (true) {
    //     cbt_Node parent;
    //     parent.id = node.id / 2;
    //     parent.depth = node.depth - 1;
    //     AABB pbounds = nodeID2AABB(parent.id, parent.depth);
    //     if (!should_split(pbounds) && node.depth > 4) {
    //         cbt_MergeNode(node);
    //     }
    // }
    // AABB pbounds = nodeID2AABB(node.id / 2, node.depth - 1);
    // if ((node.id % 2 == 0) && should_split(pbounds) && node.depth - 1 > 3) {
    //     cbt_Node pnode;
    //     pnode.id = node.id / 2;
    //     pnode.depth = node.depth - 1;
    //     // cbt_MergeNode(pnode);
    // }
    // else if(node.depth > 3) {
    //     cbt_MergeNode(node);
    // }
}