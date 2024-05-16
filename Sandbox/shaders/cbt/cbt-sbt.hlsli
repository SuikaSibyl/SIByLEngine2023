#ifndef _SRENDERER_CBT_SBT_HLSLI_   
#define _SRENDERER_CBT_SBT_HLSLI_

#include "common/geometry.hlsli"
#include "common/space_filling_curve.hlsli"

AABB nodeID2AABB(uint nodeID, uint depth) {
    const int x_level = 1 << ((depth + 2) / 3);
    const int y_level = 1 << ((depth + 1) / 3);
    const int z_level = 1 << ((depth + 0) / 3);
    const int x_shift = (depth + 2) % 3;
    const int y_shift = (depth + 1) % 3;
    const int z_shift = (depth + 0) % 3;
    const int x_index = inverse_morton3(nodeID >> x_shift) & (x_level - 1);
    const int y_index = inverse_morton3(nodeID >> y_shift) & (y_level - 1);
    const int z_index = inverse_morton3(nodeID >> z_shift) & (z_level - 1);
    const float3 index = float3(x_index, y_index, z_index);
    const float3 level = float3(x_level, y_level, z_level);
    // create the AABB bounds for the node
    AABB bounds;
    bounds.min = (index + 0) / level;
    bounds.max = (index + 1) / level;
    return bounds;
}

#endif // _SRENDERER_CBT_SBT_HLSLI_