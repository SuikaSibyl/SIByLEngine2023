#ifndef _SRENDERER_ADDON_DIFFERENTIABLE_REPARAM_HEADER_
#define _SRENDERER_ADDON_DIFFERENTIABLE_REPARAM_HEADER_

#include "../../../include/common/linear_algebra.hlsli"
#include "../../../raytracer/spt_interface.hlsli"

/**
 * 
 * @generic nb_samples: number of samples, must be larger than 2
 */
 [Differentiable]
float3 estimate_discontinuity<let nb_samples : int>(
    in_ref(Ray) rays[nb_samples],
    in_ref(GeometryHit) sis[nb_samples],
) {
    float3 ray0_p_attached = sis[0].position;
    float3 ray0_n = sis[0].geometryNormal;

    // For all the other samples (indices are not 0), 
    // try to find one has different geometryID
    uint is_ray1_hit_uint = select(HasHit(sis[1]), 1u, 0u);
    float3 ray1_p_attached = sis[1].position;
    float3 ray1_n = sis[1].geometryNormal;
    float3 ray1_d = rays[1].direction;
    for (uint i = 2; i < nb_samples; i++) {
        const bool diff = (sis[0].geometryID != sis[i].geometryID);
        const bool i_hit = HasHit(sis[i]);
        is_ray1_hit_uint = select(diff, select(i_hit, 1u, 0u), is_ray1_hit_uint);
        ray1_p_attached = select(diff, sis[i].position, ray1_p_attached);
        ray1_n = select(diff, sis[i].geometryNormal, ray1_n);
        ray1_d = select(diff, rays[i].direction, ray1_d);
    }
    // Whether the compared sample actually hit some geometry.
    const bool is_ray1_hit = is_ray1_hit_uint > 0;

    // Guess occlusion for pairs of samples
    float3 res = float3(0.f);
    
    // if only one hit: return this hit
    const bool only_hit_0 = HasHit(sis[0]) && !is_ray1_hit;
    if (only_hit_0) res = ray0_p_attached;

    const bool only_hit_1 = is_ray1_hit && (!HasHit(sis[0]));
    if (only_hit_1) res = ray1_p_attached;
    
    const bool has_two_hits = HasHit(sis[0]) && is_ray1_hit;

    // Compute occlusion between planes and hitpoints:
    // sign of dot(normal, hitpoint - hitpoint).
    // Test if the origin of the rays is on the same side as the other hit.
    const float occ_plane_0 =
        dot(ray0_n, ray1_p_attached - ray0_p_attached) *
        dot(ray0_n, rays[0].origin - ray0_p_attached);
    const float occ_plane_1 =
        dot(ray1_n, ray0_p_attached - ray1_p_attached) *
        dot(ray0_n, rays[0].origin - ray0_p_attached);

    const bool plane_0_occludes_1 = has_two_hits && (occ_plane_0 < 0.f);
    const bool plane_1_occludes_0 = has_two_hits && (occ_plane_1 < 0.f);

    const bool simple_occluder_0 = plane_0_occludes_1 && !plane_1_occludes_0;
    const bool simple_occluder_1 = plane_1_occludes_0 && !plane_0_occludes_1;
    bool plane_intersection = has_two_hits && !simple_occluder_1 && !simple_occluder_0;

    /* simple_occluder */
    if (simple_occluder_0) res = ray0_p_attached;
    if (simple_occluder_1) res = ray1_p_attached;

    /* same_normals */

    const bool same_normals = plane_intersection && abs(dot(ray0_n, ray1_n)) > 0.99f;
    plane_intersection &= !same_normals;
    if (same_normals) res = ray0_p_attached;

    /* plane_intersection */

    // Compute the intersection between 3 planes:
    // 2 planes defined by the ray intersections and
    // the normals at these points, and 1 plane containing
    // the ray directions.

    float3 N0 = ray0_n;
    float3 N1 = ray1_n;
    float3 P0 = ray0_p_attached;
    float3 P1 = ray1_p_attached;

    // Normal of the third plane, defined using
    // attached positions (this prevents bad correlations
    // between the displacement of the intersection and
    // the sampled positions)

    float3 N = cross(P0 - rays[0].origin, P1 - rays[0].origin);
    float norm_N = length(N);

    // Set a default intersection if the problem is ill-defined
    if (plane_intersection) res = ray0_p_attached;

    const bool invertible = plane_intersection && norm_N > 0.001f;

    float3x3 A = float3x3(N0, N1, N); // TODO :: check whether require transpose
    const float b0 = dot(P0, N0);
    const float b1 = dot(P1, N1);
    const float b2 = dot(rays[0].origin, N);
    float3 B = float3(b0, b1, b2);
    float3x3 invA = Inverse3x3(A);
    if (invertible) res = mul(invA, B); // TODO :: check which direction to use
    
    return res;
}

#endif // _SRENDERER_ADDON_DIFFERENTIABLE_REPARAM_HEADER_