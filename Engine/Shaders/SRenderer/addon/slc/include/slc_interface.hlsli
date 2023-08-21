#ifndef _SRENDERER_ADDON_SLC_INTERFACE_HEADER_
#define _SRENDERER_ADDON_SLC_INTERFACE_HEADER_

#include "slc_common.hlsli"

enum SLCSampleType {
    Random,     // Uniformly sample a light
    LightTree,  // Traverse the light tree once to pick a light
    LightCut,   // Use a light cut to pick multiple lights
};

struct SLCNode {
    float3 boundMin;
    float intensity;
    float3 boundMax;
    int ID;
    float4 cone; // xyz cone axis, w cone angle
};

struct SLCEvaluateConfig {
    bool useApproximateCosineBound;
    bool useLightCone;
    int distanceType;
};

SLCEvaluateConfig UnpackSLCEvaluateConfig(uint bitfield) {
    SLCEvaluateConfig config;
    config.useApproximateCosineBound = (bitfield & 1) != 0;
    config.useLightCone = (bitfield & 2) != 0;
    config.distanceType = (bitfield >> 2) & 3;
    return config;
}

/**
 * Evaluate the first child weight in a binary tree.
 * @param c0 The first child node.
 * @param c1 The second child node.
 * @param p The reference point position.
 * @param n The reference point normal.
 * @param v The reference point view direction.
 * @param config The evaluation configuration.
 * @param prob0 The output probability of the first child.
 * @return True if the evaluation is successful.
*/
bool EvaluateFirstChildWeight(
    in_ref(SLCNode) c0,
    in_ref(SLCNode) c1,
    in_ref(float3) p,
    in_ref(float3) n,
    in_ref(float3) v,
    in_ref(SLCEvaluateConfig) config,
    out_ref(float) prob0
) {
    const float c0_intensity = c0.intensity;
    const float c1_intensity = c1.intensity;

    if (c0_intensity == 0) {
        if (c1_intensity == 0) return false;
        prob0 = 0; return true;
    }
    else if (c1_intensity == 0) {
        prob0 = 1; return true;
    }
    
    const float3 c0_boundMin = c0.boundMin;
    const float3 c0_boundMax = c0.boundMax;
    const float3 c1_boundMin = c1.boundMin;
    const float3 c1_boundMax = c1.boundMax;

    // Compute the weights
    float geom0 = 1;
    float geom1 = 1;

    if (config.useApproximateCosineBound) {
        geom0 = GeomTermBoundApproximate(p, n, c0_boundMin, c0_boundMax);
        geom1 = GeomTermBoundApproximate(p, n, c1_boundMin, c1_boundMax);
    }
    else {
        geom0 = GeomTermBound(p, n, c0_boundMin, c0_boundMax);
        geom1 = GeomTermBound(p, n, c1_boundMin, c1_boundMax);
    }

    if (config.useLightCone) {
        const float3 c0r_boundMin = 2 * p - c0_boundMax;
        const float3 c0r_boundMax = 2 * p - c0_boundMin;
        const float3 c1r_boundMin = 2 * p - c1_boundMax;
        const float3 c1r_boundMax = 2 * p - c1_boundMin;

        float cos0 = 1;
        float cos1 = 1;
        
        if (config.useApproximateCosineBound) {
            cos0 = GeomTermBoundApproximate(p, c0.cone.xyz, c0r_boundMin, c0r_boundMax);
            cos1 = GeomTermBoundApproximate(p, c1.cone.xyz, c1r_boundMin, c1r_boundMax);
        }
        else {
            cos0 = GeomTermBound(p, c0.cone.xyz, c0r_boundMin, c0r_boundMax);
            cos1 = GeomTermBound(p, c1.cone.xyz, c1r_boundMin, c1r_boundMax);
        }

        geom0 *= max(0.f, cos(max(0.f, acos(cos0) - c0.cone.w)));
        geom1 *= max(0.f, cos(max(0.f, acos(cos1) - c1.cone.w)));
    }

    if (geom0 + geom1 == 0)
        return false;

    if (geom0 == 0) {
        prob0 = 0;
        return true;
    }
    else if (geom1 == 0) {
        prob0 = 1;
        return true;
    }

    const float intensGeom0 = c0_intensity * geom0;
    const float intensGeom1 = c1_intensity * geom1;

    float l2_min0;
    float l2_min1;
    l2_min0 = SquaredDistanceToClosestPoint(p, c0_boundMin, c0_boundMax);
    l2_min1 = SquaredDistanceToClosestPoint(p, c1_boundMin, c1_boundMax);


    if (config.distanceType == 0) {
        if (l2_min0 < WidthSquared(c0_boundMin, c0_boundMax) 
         || l2_min1 < WidthSquared(c1_boundMin, c1_boundMax)) {
            prob0 = intensGeom0 / (intensGeom0 + intensGeom1);
        }
        else {
            float w_max0 = normalizedWeights(l2_min0, l2_min1, intensGeom0, intensGeom1);
            prob0 = w_max0; // closest point
        }
    }
    else if (config.distanceType == 1) {
        const float3 l0 = 0.5 * (c0_boundMin + c0_boundMax) - p;
        const float3 l1 = 0.5 * (c1_boundMin + c1_boundMax) - p;
        const float w_max0 = normalizedWeights(max(0.001, dot(l0, l0)), max(0.001, dot(l1, l1)), intensGeom0, intensGeom1);
        prob0 = w_max0; // closest point
    }
    else if (config.distanceType == 2) {
        // avg weight of minmax (used in the paper)
        const float l2_max0 = SquaredDistanceToFarthestPoint(p, c0_boundMin, c0_boundMax);
        const float l2_max1 = SquaredDistanceToFarthestPoint(p, c1_boundMin, c1_boundMax);
        const float w_max0 = l2_min0 == 0 && l2_min1 == 0 ? intensGeom0 / (intensGeom0 + intensGeom1) : normalizedWeights(l2_min0, l2_min1, intensGeom0, intensGeom1);
        const float w_min0 = normalizedWeights(l2_max0, l2_max1, intensGeom0, intensGeom1);
        prob0 = 0.5 * (w_max0 + w_min0);
    }
    if (config.distanceType == 3) {
        prob0 = intensGeom0 / (intensGeom0 + intensGeom1);
    }
    return true;
}

/**
* Traverse the light tree to pick a light.
* @param nid The node ID to start the traversal from.
* @param LeafStartIndex The index of the first leaf node.
* @param p The reference point position.
* @param n The reference point normal.
* @param v The reference point view direction.
* @param rnd The random number in [0, 1).
* @param nprob The normalized probability of the picked light.
* @param config The evaluation configuration.
* @param nodeBuffer The light tree node buffer.
*/
inline int TraverseLightTree(
    int nid,
    int LeafStartIndex,
    in_ref(float3) p,
    in_ref(float3) n,
    in_ref(float3) v,
    in_ref(float) rnd,
    inout_ref(double) nprob,
    in_ref(SLCEvaluateConfig) config,
    StructuredBuffer<SLCNode> nodeBuffer
) {
    nprob = 1.;
    while (nid < LeafStartIndex) {
        int c0_id = nid << 1;   // left child
        int c1_id = c0_id + 1;  // right child

        float prob0;
        const SLCNode c0 = nodeBuffer[c0_id];
        const SLCNode c1 = nodeBuffer[c1_id];
        if (EvaluateFirstChildWeight(c0, c1, p, n, v, config, prob0)) {
            if (rnd < prob0) {
                nid = c0_id;
                rnd /= prob0;
                nprob *= double(prob0);
            }
            else {
                nid = c1_id;
                rnd = (rnd - prob0) / (1 - prob0);
                nprob *= double(1 - prob0);
            }
        }
        else {
            // dead branch and thus invalid sample
            nid = -1;
            break;
        }
    }
    return nid;
}

#endif // _SRENDERER_ADDON_SLC_INTERFACE_HEADER_