#ifndef _SRENDERER_ADDON_VXGUIDING_TREE_SHARED_HEADER_
#define _SRENDERER_ADDON_VXGUIDING_TREE_SHARED_HEADER_

#include "../../../include/common/cpp_compatible.hlsli"
#include "../../../include/common/packing.hlsli"

struct TreeConstrIndirectArgs {
    int3 dispatch_leaf;
    uint numValidVPLs;
    int3 dispatch_internal;
    uint padding0;
    int3 dispatch_node;
    uint padding1;
    int4 draw_rects;
};

struct TreeNode {
    // 16
    uint2 aabbMin;
    uint2 aabbMax;
    // 16
    float intensity;
    uint flag;           // got node already processed?
    uint16_t parent_idx; // parent node
    uint16_t left_idx;   // index of left  child node
    uint16_t right_idx;  // index of right child node
    uint16_t vx_idx;     // == 0xFFFF if internal node.
    // member functions
    inline float3 getAABBMin() { return unpackFloat3(aabbMin); }
    inline float3 getAABBMax() { return unpackFloat3(aabbMax); }
    mutating inline void setAABBMin(in_ref(float3) min) { packFloat3(min, aabbMin); }
    mutating inline void setAABBMax(in_ref(float3) max) { packFloat3(max, aabbMax); }
};

#endif // _SRENDERER_ADDON_VXGUIDING_TREE_SHARED_HEADER_