#ifndef _SRENDERER_ADDON_SEMINEE_INTERFACE_HEADER_
#define _SRENDERER_ADDON_SEMINEE_INTERFACE_HEADER_

#include "../../include/common/cpp_compatible.hlsli"
#include "../../include/common/packing.hlsli"

struct TreeConstrIndirectArgs {
    int3 dispatch_leaf;
    uint numValidVPLs;
    int3 dispatch_internal;
    uint padding0;
    int3 dispatch_node;
    uint padding1;
    int4 draw_rects;
};

uint GetVPLIndex(in_ref(int2) pixel, in_ref(int2) resolution) {
    return pixel.x * resolution.y + pixel.y;
}

struct TreeNode {
    uint parent_idx; // parent node
    uint left_idx;   // index of left  child node
    uint right_idx;  // index of right child node
    uint vpl_idx;    // == 0xFFFFFFFF if internal node.
    uint flag;       // got node already processed?
};

struct VPLMerge {
    float2 ApproxScore;
};

struct VPLData {
    uint2 posW;
    uint2 normW;
    // 16
    uint2 aabbMin;
    uint2 aabbMax;
    // 16
    uint2 rad;
    uint2 var;
    // 16
    int id;
    int idChild1;
    int idChild2;
    int numVPLSubTree;
    // 16
    inline bool isInvalid() { return id == -1; }
    inline float3 getPosW() { return unpackFloat3(posW); }
    inline float3 getNormW() { return unpackFloat3(normW); }
    inline float3 getColor() { return unpackFloat3(rad); }
    inline float getIntensity() { return unpackFloatLow(rad.y); }
    inline float3 getAABBMin() { return unpackFloat3(aabbMin); }
    inline float3 getAABBMax() { return unpackFloat3(aabbMax); }
    inline float3 getVariance() { return unpackFloat3(var); }
    inline float getEarlyStop() { return unpackFloatLow(var.y); }
    mutating inline void setInvalid() { id = -1; }
    mutating inline void setPosW(float3 p) { packFloat3(p, posW); }
    mutating inline void setNormW(float3 n) { packFloat3(n, normW); }
    mutating inline void setColor(float3 c) { packFloat3(c, rad); }
    mutating inline void setIntensity(float i) { packFloatLow(i, rad.y); }
    mutating inline void setAABBMin(float3 min) { packFloat3(min, aabbMin); }
    mutating inline void setAABBMax(float3 max) { packFloat3(max, aabbMax); }
    mutating inline void setVariance(float3 variance) { packFloat3(variance, var); }
    mutating inline void setEarlyStop(float f) { packFloatLow(f, var.y); }
};

struct TreeApproxParams {
    float minNormalScore;
    float maxNormalZStd;
    float pad2;
    float pad3;
};

#endif // _SRENDERER_ADDON_SEMINEE_INTERFACE_HEADER_