#ifndef _SRENDERER_ADDON_SLC_UTILS_HEADER_
#define _SRENDERER_ADDON_SLC_UTILS_HEADER_

#include "../../../include/common/cpp_compatible.hlsli"
#include "../../../include/common/geometry.hlsli"

/**
* Compute the maximum distance along the direction dir to the bounding box bound.
* @param p The origin of the ray.
* @param dir The direction of the ray.
* @param bound The bounding box.
*/
float MaxDistAlong(in_ref(float3) p, in_ref(float3) dir, in_ref(AABB) bound) {
    const float3 dir_p = dir * p;
    const float3 mx0 = dir * bound.min - dir_p;
    const float3 mx1 = dir * bound.max - dir_p;
    return max(mx0[0], mx1[0]) + max(mx0[1], mx1[1]) + max(mx0[2], mx1[2]);
}

/**
* Compute the minimum distance along the direction dir to the bounding box bound.
* @param p The origin of the ray.
* @param dir The direction of the ray.
* @param bound The bounding box.
*/
float AbsMinDistAlong(in_ref(float3) p, in_ref(float3) dir, in_ref(AABB) bound) {
    bool hasPositive = false;
    bool hasNegative = false;
    float a = dot(dir, float3(bound.min.x, bound.min.y, bound.min.z) - p);
    float b = dot(dir, float3(bound.min.x, bound.min.y, bound.max.z) - p);
    float c = dot(dir, float3(bound.min.x, bound.max.y, bound.min.z) - p);
    float d = dot(dir, float3(bound.min.x, bound.max.y, bound.max.z) - p);
    float e = dot(dir, float3(bound.max.x, bound.min.y, bound.min.z) - p);
    float f = dot(dir, float3(bound.max.x, bound.min.y, bound.max.z) - p);
    float g = dot(dir, float3(bound.max.x, bound.max.y, bound.min.z) - p);
    float h = dot(dir, float3(bound.max.x, bound.max.y, bound.max.z) - p);
    hasPositive = a > 0 || b > 0 || c > 0 || d > 0 || e > 0 || f > 0 || g > 0 || h > 0;
    hasNegative = a < 0 || b < 0 || c < 0 || d < 0 || e < 0 || f < 0 || g < 0 || h < 0;
    if (hasPositive && hasNegative) return 0.f;
    else return min(min(min(abs(a), abs(b)), min(abs(c), abs(d))), min(min(abs(e), abs(f)), min(abs(g), abs(h))));
}

void CoordinateSystem_(in_ref(float3) v1, out_ref(float3) v2, out_ref(float3) v3) {
    if (abs(v1.x) > abs(v1.y)) v2 = float3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    else v2 = float3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    v3 = normalize(cross(v1, v2));
}

float GeomTermBound(in_ref(float3) p, in_ref(float3) N, in_ref(AABB) bound) {
    const float nrm_max = MaxDistAlong(p, N, bound);
    if (nrm_max <= 0) return 0.0f;
    float3 T, B;
    CoordinateSystem_(N, T, B);
    const float y_amin = AbsMinDistAlong(p, T, bound);
    const float z_amin = AbsMinDistAlong(p, B, bound);
    const float hyp2 = y_amin * y_amin + z_amin * z_amin + nrm_max * nrm_max;
    return nrm_max * rsqrt(hyp2);
}

float GeomTermBoundApproximate(in_ref(float3) p, in_ref(float3) N, in_ref(AABB) bound) {
    const float nrm_max = MaxDistAlong(p, N, bound);
    if (nrm_max <= 0) return 0.0f;
    const float3 d = min(max(p, bound.min), bound.max) - p;
    const float3 tng = d - dot(d, N) * N;
    const float hyp2 = dot(tng, tng) + nrm_max * nrm_max;
    return nrm_max * rsqrt(hyp2);
}

float WidthSquared(in_ref(AABB) bound) {
    const float3 d = bound.max - bound.min;
    return d.x * d.x + d.y * d.y + d.z * d.z;
}

float SquaredDistanceToClosestPoint(in_ref(float3) p, in_ref(AABB) bound) {
    const float3 d = min(max(p, bound.min), bound.max) - p;
    return dot(d, d);
}

float SquaredDistanceToFarthestPoint(in_ref(float3) p, in_ref(AABB) bound) {
    const float3 d = max(abs(bound.min - p), abs(bound.max - p));
    return dot(d, d);
}

float InvMaxDistance(float x) {
    if (isinf(x)) return 0;
    else return 1. / x;
}

// vec4 InvMinDistanceFixing(vec4 v) {
//     const vec4 is_zero = vec4(v.x == 0, v.y == 0, v.z == 0, v.w == 0);
//     if (is_zero == vec4(0)) return 1. / v;
//     else return is_zero;
// }

// vec4 normalizeWeights(vec4 w) {
//     const float sum = dot(w, vec4(1.0));
//     if (sum == 0) return vec4(0);
//     return w / sum;
// }

struct GeometryTermSetting {
    bool useApproximateCosineBound;
    bool useLightCone;
};

float ComputeGeometryTerm(
    in_ref(float3) position,
    in_ref(float3) normal,
    in_ref(AABB) bound,
    in_ref(GeometryTermSetting) setting
) {
    float geom = setting.useApproximateCosineBound
                     ? GeomTermBoundApproximate(position, normal, bound)
                     : GeomTermBound(position, normal, bound);
    return geom;          
}

enum SLCDistanceMode {
    kAverageMinmax,
};

// vec4 ComputeNodeWeights_QuadTree(
//     in const vec3 p,
//     in const vec3 N,
//     in const vec4 intensity,
//     in const SLCNode node_0,
//     in const SLCNode node_1,
//     in const SLCNode node_2,
//     in const SLCNode node_3,
//     in const uint slc_mode
// ) {
//     vec4 weights;

//     bool useApproximateCosineBound = false;

//     const vec4 isValidNode = vec4(
//         node_0.bound.min.x == k_inf ? 0 : 1,
//         node_1.bound.min.x == k_inf ? 0 : 1,
//         node_2.bound.min.x == k_inf ? 0 : 1,
//         node_3.bound.min.x == k_inf ? 0 : 1
//     );

//     // Geoterm
//     vec4 geom;
//     if (useApproximateCosineBound) {
//         geom[0] = GeomTermBoundApproximate(p, N, node_0.bound);
//         geom[1] = GeomTermBoundApproximate(p, N, node_1.bound);
//         geom[2] = GeomTermBoundApproximate(p, N, node_2.bound);
//         geom[3] = GeomTermBoundApproximate(p, N, node_3.bound);
//     } else {
//         geom[0] = GeomTermBound(p, N, node_0.bound);
//         geom[1] = GeomTermBound(p, N, node_1.bound);
//         geom[2] = GeomTermBound(p, N, node_2.bound);
//         geom[3] = GeomTermBound(p, N, node_3.bound);
//     }

//     // Lightcone
//     if ((slc_mode & 0x1) != 0) { // if use light cone
//         const AABB c0r_bound = AABB(2 * p - node_0.bound.max, 2 * p - node_0.bound.min);
//         const AABB c1r_bound = AABB(2 * p - node_1.bound.max, 2 * p - node_1.bound.min);
//         const AABB c2r_bound = AABB(2 * p - node_2.bound.max, 2 * p - node_2.bound.min);
//         const AABB c3r_bound = AABB(2 * p - node_3.bound.max, 2 * p - node_3.bound.min);

//         vec4 coss;
//         if (useApproximateCosineBound) {
//             coss[0] = GeomTermBoundApproximate(p, node_0.cone.xyz, c0r_bound);
//             coss[1] = GeomTermBoundApproximate(p, node_1.cone.xyz, c1r_bound);
//             coss[2] = GeomTermBoundApproximate(p, node_2.cone.xyz, c2r_bound);
//             coss[3] = GeomTermBoundApproximate(p, node_3.cone.xyz, c3r_bound);
//         } else {
//             coss[0] = GeomTermBound(p, node_0.cone.xyz, c0r_bound);
//             coss[1] = GeomTermBound(p, node_1.cone.xyz, c1r_bound);
//             coss[2] = GeomTermBound(p, node_2.cone.xyz, c2r_bound);
//             coss[3] = GeomTermBound(p, node_3.cone.xyz, c3r_bound);
//         }
//         const vec4 coner = vec4(node_0.cone.w, node_1.cone.w, node_2.cone.w, node_3.cone.w);
//         geom *= max(vec4(0.f), cos(max(vec4(0.f), acos(coss) - coner)));
//     }

//     for (int i = 0; i < 4; ++i) {
//         if (isValidNode[i] == 0) geom[i] = 0;
//     }

//     // Distance
//     const vec4 l2_min = vec4(
//         SquaredDistanceToClosestPoint(p, node_0.bound),
//         SquaredDistanceToClosestPoint(p, node_1.bound),
//         SquaredDistanceToClosestPoint(p, node_2.bound),
//         SquaredDistanceToClosestPoint(p, node_3.bound)
//     );
//     const vec4 il2_min = InvMinDistanceFixing(l2_min);

//     const vec4 intensGeom = intensity * geom;
//     // const vec4 intensGeom = intensity;

//     // if (l2_min_0 < WidthSquared(node_0.bound)
//     //  || l2_min_1 < WidthSquared(node_1.bound)
//     //  || l2_min_2 < WidthSquared(node_2.bound)
//     //  || l2_min_3 < WidthSquared(node_3.bound))
//     // {
//     //     // prob0 = intensGeom0 / (intensGeom0 + intensGeom1);
//     // }
//     // else
//     // {
//     //     // float w_max0 = normalizedWeights(l2_min0, l2_min1, intensGeom0, intensGeom1);
//     //     // prob0 = w_max0;	// closest point
//     // }
//     const vec4 il2_max = vec4(
//         InvMaxDistance(SquaredDistanceToFarthestPoint(p, node_0.bound)),
//         InvMaxDistance(SquaredDistanceToFarthestPoint(p, node_1.bound)),
//         InvMaxDistance(SquaredDistanceToFarthestPoint(p, node_2.bound)),
//         InvMaxDistance(SquaredDistanceToFarthestPoint(p, node_3.bound))
//     );

//     const vec4 w_max = normalizeWeights(il2_min * intensGeom);
//     const vec4 w_min = normalizeWeights(il2_max * intensGeom);
//     weights = 0.5 * (w_max + w_min);

//     return normalizeWeights(weights);
// }

#endif // _SRENDERER_ADDON_SLC_UTILS_HEADER_