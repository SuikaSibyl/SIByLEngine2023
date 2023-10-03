#include "../../include/common/geometry.hlsli"
#include "../../include/common/space_filling_curve.hlsli"
#include "sst-interface.hlsli"

[[vk::push_constant]]
cbuffer PushConstantBuffer {
    TreeApproxParams gApproxParams;
};

RWStructuredBuffer<TreeNode> u_Nodes;
RWStructuredBuffer<VPLMerge> u_Merge;
RWStructuredBuffer<VPLData> u_VPLData;

const StructuredBuffer<TreeConstrIndirectArgs> u_ConstrIndirectArgs;

[shader("compute")]
[numthreads(256, 1, 1)]
void ComputeMain(uint3 DTid: SV_DispatchThreadID) {
    uint numVPLs = u_ConstrIndirectArgs[0].numValidVPLs;
    const int numInternalNodes = int(numVPLs - 1);
    const int idx = int(DTid.x + numInternalNodes);

    // Get first parent
    uint parent = u_Nodes[idx].parent_idx;

    // Get node and vpl index for first child
    int lhs_node_id = u_Nodes[parent].left_idx;
    int lhs_vpl_id  = u_Nodes[lhs_node_id].vpl_idx;

    // Load first child data
    VPLData  lhs       = u_VPLData[lhs_vpl_id];
    VPLMerge lhs_merge = u_Merge[lhs_vpl_id];

    while (parent != 0xFFFFFFFF) { // invalid index
        uint old;
        InterlockedCompareExchange(u_Nodes[parent].flag, 0u, 1u, old);
        if (old == 0) {
            // this is the first thread entered here.
            // wait the other thread from the other child node.
            return;
        }
        
        // Get child node indices
        const uint lidx = u_Nodes[parent].left_idx;
        const uint ridx = u_Nodes[parent].right_idx;

        // Determine vpl/node indices for right-hand-side
        const uint rhs_node_id = (lhs_node_id != ridx) ? ridx : lidx;
        const uint rhs_vpl_id  = u_Nodes[rhs_node_id].vpl_idx;
        
        // Load data of second child
        VPLData  rhs       = u_VPLData[rhs_vpl_id];
        VPLMerge rhs_merge = u_Merge[rhs_vpl_id];

        // Get merge vpl id
        const uint parent_vpl_id   = u_Nodes[parent].vpl_idx;

        // Merge nodes
        VPLData merged;
        merged.posW     = float3(0);
        merged.normW    = float3(0);
        merged.id       = 0;
        merged.idChild1 = 0;
        merged.idChild2 = 0;
        merged.numVPLSubTree = 0;
        
        const float lhsIntensity = lhs.getIntensity();
        const float rhsIntensity = rhs.getIntensity();
        const float totalIntensity = lhsIntensity + rhsIntensity;
        const float alpha = totalIntensity > 0.f ? lhsIntensity / totalIntensity : 0.5f;

        merged.setIntensity(totalIntensity);
        const float3 CombinedColor = lhs.getColor() + rhs.getColor();
        merged.setColor(CombinedColor);

        const float3 normW = normalize(lerp(lhs.getNormW(), rhs.getNormW(), 1.f - alpha));
        merged.setNormW(normW);

        const float3 aabbMin = min(lhs.getAABBMin(), rhs.getAABBMin());
        const float3 aabbMax = max(lhs.getAABBMax(), rhs.getAABBMax());
        merged.setAABBMin(aabbMin);
        merged.setAABBMax(aabbMax);

        merged.id = parent_vpl_id;
        merged.idChild1 = lhs.id;
        merged.idChild2 = rhs.id;
        merged.numVPLSubTree = lhs.numVPLSubTree + rhs.numVPLSubTree + 2;

        // Compute Variance and Mean
        float3 mean, variance;
        float weight;
        const float3 lhsPos = lhs.getPosW();
        const float3 rhsPos = rhs.getPosW();
        iterativePlaneWeightedMeanAndVariance(normW, lhsPos, rhsPos, lhs.getVariance(), rhs.getVariance(), lhsIntensity, rhsIntensity, mean, variance, weight);
        merged.setPosW(mean);
        merged.setVariance(variance);

        // Compute Approximation
        VPLMerge merge_data;
        merge_data.ApproxScore   = float2(0.f);
        merge_data.ApproxScore.x = lhs_merge.ApproxScore.x * rhs_merge.ApproxScore.x * computeNormalScore(lhs.getNormW(), rhs.getNormW());
        merge_data.ApproxScore.y = sqrt(variance.z);

        const float earlyStop = evalScores(merge_data.ApproxScore, gApproxParams) ? 1.f : 0.f;
        merged.setEarlyStop(earlyStop);

        // Write merged vpl data
        u_VPLData[parent_vpl_id] = merged;
        u_Merge[parent_vpl_id]   = merge_data;

        // Save merged vpl for next round
        lhs       = merged;
        lhs_merge = merge_data;

        lhs_vpl_id  = parent_vpl_id;
        lhs_node_id = parent;

        // look for the next parent...
        parent = u_Nodes[parent].parent_idx;
    }
}

void iterativePlaneWeightedMeanAndVariance(
    float3 N, float3 m1, float3 m2, float3 v1, float3 v2, 
    float w1, float w2, 
    inout float3 m, inout float3 v, inout float w)
{
    w = w1 + w2;

    if (w != 0.f) {
        // Update mean
        m = (w1 * m1 + w2 * m2) / w;

        const float3x3 R = getRotationMatrixFromAToB(float3(0.f, 0.f, 1.f), N);
        const float3 Pm  = mul(m,  R);
        const float3 Pm1 = mul(m1, R);
        const float3 Pm2 = mul(m2, R);

        // Update variance
        float3 d1 = Pm1 - Pm;
        float3 d2 = Pm2 - Pm;
        v = (w1 * (v1 + d1 * d1) + w2 * (v2 + d2 * d2)) / w;
    }
    else {
        m = (m1 + m2) * 0.5f;
        v.x = v.y = v.z = 0.f;
    }
}

float computeNormalScore(const float3 n1, const float3 n2) {
    return max(0.f, dot(n1, n2));
}

bool evalScores(const float2 approxScore, const TreeApproxParams approxParams) {
    // Is our approximation good enough? Return true if yes!
    return
        (approxScore.x >= approxParams.minNormalScore) &&   // Normal score
        (approxScore.y <= approxParams.maxNormalZStd);      // Normal Z-Std
}