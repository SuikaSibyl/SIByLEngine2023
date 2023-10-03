#ifndef _SRENDERER_ADDON_SST_COMMON_HEADER_
#define _SRENDERER_ADDON_SST_COMMON_HEADER_

#include "../../include/common/packing.hlsli"

struct VPLData {
    float3 posW;
    float aabbMax_x;
    float3 normW;
    float aabbMax_y;
    float3 color;
    float intensity;
    // 16
    float3 aabbMin;
    float aabbMax_z;
    // 16
    float3 variance;
    float earlyStop;
    // 16
    int id;
    int idChild1;
    int idChild2;
    int numVPLSubTree;
    // 16
    inline bool isInvalid() { return id == -1; }
    inline float3 getPosW() { return posW; }
    inline float3 getNormW() { return normW; }
    inline float3 getColor() { return color; }
    inline float getIntensity() { return intensity; }
    inline float3 getAABBMin() { return aabbMin; }
    inline float3 getAABBMax() { return float3(aabbMax_x, aabbMax_y, aabbMax_z); }
    inline float3 getVariance() { return variance; }
    inline float getEarlyStop() { return earlyStop; }
    mutating inline void setInvalid() { id = -1; }
    mutating inline void setPosW(float3 p) { posW = p; }
    mutating inline void setNormW(float3 n) { normW = n; }
    mutating inline void setColor(float3 c) { color = c; }
    mutating inline void setIntensity(float i) { intensity = i; }
    mutating inline void setAABBMin(float3 min) { aabbMin = min; }
    mutating inline void setAABBMax(float3 max) {
        aabbMax_x = max.x;
        aabbMax_y = max.y;
        aabbMax_z = max.z;
    }
    mutating inline void setVariance(float3 vari) { variance = vari; }
    mutating inline void setEarlyStop(float f) { earlyStop = f; }
};

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
    uint parent_idx; // parent node
    uint left_idx;   // index of left  child node
    uint right_idx;  // index of right child node
    uint vpl_idx;    // == 0xFFFFFFFF if internal node.
    uint flag;       // got node already processed?
};

struct VPLMerge {
    float2 ApproxScore;
};

struct TreeApproxParams {
    float minNormalScore;
    float maxNormalZStd;
};

#endif // _SRENDERER_ADDON_SST_COMMON_HEADER_