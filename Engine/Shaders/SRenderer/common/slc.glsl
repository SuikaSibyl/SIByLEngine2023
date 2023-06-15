#ifndef _STOCHASTIC_LIGHT_CUT_HEADER_
#define _STOCHASTIC_LIGHT_CUT_HEADER_

#include "../../Utility/geometry.h"

/**
* A Node for Stochastic Lightcuts Hierarchy
* @param bound: bounding box of the node
* @param cone: normal cone of the node (xyz: direction, w: angle)
*/
struct SLCNode {
    AABB bound;
    vec4 cone;
};

void Swap(inout int first, inout int second) {
	int temp = first;
	first = second;
	second = temp;
}

float MaxDistAlong(in const vec3 p, in const vec3 dir, in const AABB bound) {
	const vec3 dir_p = dir * p;
	const vec3 mx0 = dir * bound.min - dir_p;
	const vec3 mx1 = dir * bound.max - dir_p;
	return max(mx0[0], mx1[0]) + max(mx0[1], mx1[1]) + max(mx0[2], mx1[2]);
}

void CoordinateSystem_(in const vec3 v1, out vec3 v2, out vec3 v3) {
	if (abs(v1.x) > abs(v1.y)) v2 = vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
	else v2 = vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
	v3 = normalize(cross(v1, v2));
}

float AbsMinDistAlong(in const vec3 p, in const vec3 dir, in const AABB bound) {
	bool hasPositive = false;
	bool hasNegative = false;
	float a = dot(dir, vec3(bound.min.x, bound.min.y, bound.min.z) - p);
	float b = dot(dir, vec3(bound.min.x, bound.min.y, bound.max.z) - p);
	float c = dot(dir, vec3(bound.min.x, bound.max.y, bound.min.z) - p);
	float d = dot(dir, vec3(bound.min.x, bound.max.y, bound.max.z) - p);
	float e = dot(dir, vec3(bound.max.x, bound.min.y, bound.min.z) - p);
	float f = dot(dir, vec3(bound.max.x, bound.min.y, bound.max.z) - p);
	float g = dot(dir, vec3(bound.max.x, bound.max.y, bound.min.z) - p);
	float h = dot(dir, vec3(bound.max.x, bound.max.y, bound.max.z) - p);
	hasPositive = a > 0 || b > 0 || c > 0 || d > 0 || e > 0 || f > 0 || g > 0 || h > 0;
	hasNegative = a < 0 || b < 0 || c < 0 || d < 0 || e < 0 || f < 0 || g < 0 || h < 0;
	if (hasPositive && hasNegative) return 0.f;
	else return min(min(min(abs(a), abs(b)), min(abs(c), abs(d))), min(min(abs(e), abs(f)), min(abs(g), abs(h))));
}

float GeomTermBound(in const vec3 p, in const vec3 N, in const AABB bound) {
	const float nrm_max = MaxDistAlong(p, N, bound);
	if (nrm_max <= 0) return 0.0f;
	vec3 T, B;
	CoordinateSystem_(N, T, B);
	const float y_amin = AbsMinDistAlong(p, T, bound);
	const float z_amin = AbsMinDistAlong(p, B, bound);
	const float hyp2 = y_amin * y_amin + z_amin * z_amin + nrm_max * nrm_max;
	return nrm_max * inversesqrt(hyp2);
}

float GeomTermBoundApproximate(in const vec3 p, in const vec3 N, in const AABB bound) {
	const float nrm_max = MaxDistAlong(p, N, bound);
	if (nrm_max <= 0) return 0.0f;
	const vec3 d = min(max(p, bound.min), bound.max) - p;
	const vec3 tng = d - dot(d, N) * N;
	const float hyp2 = dot(tng, tng) + nrm_max * nrm_max;
	return nrm_max * inversesqrt(hyp2);
}

float WidthSquared(in const AABB bound) {
	const vec3 d = bound.max - bound.min;
	return d.x*d.x + d.y*d.y + d.z*d.z;
}

float SquaredDistanceToClosestPoint(in const vec3 p, in const AABB bound) {
	const vec3 d = min(max(p, bound.min), bound.max) - p;
	return dot(d, d);
}

float SquaredDistanceToFarthestPoint(in const vec3 p, in const AABB bound) {
	const vec3 d = max(abs(bound.min - p), abs(bound.max - p));
	return dot(d, d);
}

float InvMaxDistance(float x) {
    if(isinf(x)) return 0;
    else return 1. / x;
}

vec4 InvMinDistanceFixing(vec4 v) {
    const vec4 is_zero = vec4(v.x==0, v.y==0, v.z==0, v.w==0);
    if(is_zero == vec4(0)) return 1. / v;
    else return is_zero;
}

vec4 normalizeWeights(vec4 w) {
    const float sum = dot(w, vec4(1.0));
    if(sum == 0) return vec4(0);
    return w / sum;
}

float CosTest(in const vec3 p, in const vec3 N, in const AABB bound) {
    const vec3 mid = bound.min + 0.5 * (bound.max - bound.min);
    const vec3 d = normalize(mid - p);
    return dot(d, N);
}

vec4 ComputeNodeWeights_QuadTree(
    in const vec3 p,
    in const vec3 N,
    in const vec4 intensity,
    in const SLCNode node_0,
    in const SLCNode node_1,
    in const SLCNode node_2,
    in const SLCNode node_3,
    in const uint slc_mode
) {
    vec4 weights;

    bool useApproximateCosineBound = false;

    const vec4 isValidNode = vec4(
        node_0.bound.min.x == k_inf ? 0 : 1,
        node_1.bound.min.x == k_inf ? 0 : 1,
        node_2.bound.min.x == k_inf ? 0 : 1,
        node_3.bound.min.x == k_inf ? 0 : 1
    );

    // Geoterm
    vec4 geom;
	if (useApproximateCosineBound) {
		geom[0] = GeomTermBoundApproximate(p, N, node_0.bound);
		geom[1] = GeomTermBoundApproximate(p, N, node_1.bound);
		geom[2] = GeomTermBoundApproximate(p, N, node_2.bound);
		geom[3] = GeomTermBoundApproximate(p, N, node_3.bound);
	} else {
		geom[0] = GeomTermBound(p, N, node_0.bound);
		geom[1] = GeomTermBound(p, N, node_1.bound);
		geom[2] = GeomTermBound(p, N, node_2.bound);
		geom[3] = GeomTermBound(p, N, node_3.bound);
	}

    // Lightcone
    if((slc_mode & 0x1) != 0) { // if use light cone
        const AABB c0r_bound = AABB(2 * p - node_0.bound.max, 2 * p - node_0.bound.min);
        const AABB c1r_bound = AABB(2 * p - node_1.bound.max, 2 * p - node_1.bound.min);
        const AABB c2r_bound = AABB(2 * p - node_2.bound.max, 2 * p - node_2.bound.min);
        const AABB c3r_bound = AABB(2 * p - node_3.bound.max, 2 * p - node_3.bound.min);

        vec4 coss;
        if (useApproximateCosineBound) {
            coss[0] = GeomTermBoundApproximate(p, node_0.cone.xyz, c0r_bound);
            coss[1] = GeomTermBoundApproximate(p, node_1.cone.xyz, c1r_bound);
            coss[2] = GeomTermBoundApproximate(p, node_2.cone.xyz, c2r_bound);
            coss[3] = GeomTermBoundApproximate(p, node_3.cone.xyz, c3r_bound);
        } else {
            coss[0] = GeomTermBound(p, node_0.cone.xyz, c0r_bound);
            coss[1] = GeomTermBound(p, node_1.cone.xyz, c1r_bound);
            coss[2] = GeomTermBound(p, node_2.cone.xyz, c2r_bound);
            coss[3] = GeomTermBound(p, node_3.cone.xyz, c3r_bound);
        }
        const vec4 coner = vec4(node_0.cone.w, node_1.cone.w, node_2.cone.w, node_3.cone.w);
        geom *= max(vec4(0.f), cos(max(vec4(0.f), acos(coss) - coner)));
    }
    
    for(int i=0;i<4;++i) {
        if(isValidNode[i] == 0) geom[i] = 0;
    }
    
    // Distance
	const vec4 l2_min = vec4(
        SquaredDistanceToClosestPoint(p, node_0.bound),
        SquaredDistanceToClosestPoint(p, node_1.bound),
        SquaredDistanceToClosestPoint(p, node_2.bound),
        SquaredDistanceToClosestPoint(p, node_3.bound)
    );
    const vec4 il2_min = InvMinDistanceFixing(l2_min);
    
    const vec4 intensGeom = intensity * geom;
    // const vec4 intensGeom = intensity;

    // if (l2_min_0 < WidthSquared(node_0.bound)
    //  || l2_min_1 < WidthSquared(node_1.bound)
    //  || l2_min_2 < WidthSquared(node_2.bound)
    //  || l2_min_3 < WidthSquared(node_3.bound))
    // {
    //     // prob0 = intensGeom0 / (intensGeom0 + intensGeom1);
    // }
    // else
    // {
    //     // float w_max0 = normalizedWeights(l2_min0, l2_min1, intensGeom0, intensGeom1);
    //     // prob0 = w_max0;	// closest point
    // }
    const vec4 il2_max = vec4(
        InvMaxDistance(SquaredDistanceToFarthestPoint(p, node_0.bound)),
        InvMaxDistance(SquaredDistanceToFarthestPoint(p, node_1.bound)),
        InvMaxDistance(SquaredDistanceToFarthestPoint(p, node_2.bound)),
        InvMaxDistance(SquaredDistanceToFarthestPoint(p, node_3.bound))
    );

    const vec4 w_max = normalizeWeights(il2_min * intensGeom);
    const vec4 w_min = normalizeWeights(il2_max * intensGeom);
    weights = 0.5 * (w_max + w_min);

    return normalizeWeights(weights);
}

#endif