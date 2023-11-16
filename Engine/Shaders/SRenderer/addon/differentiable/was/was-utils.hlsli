#ifndef _SRENDERER_DIFFERENTIABLE_WARP_UTILS_HEADER_
#define _SRENDERER_DIFFERENTIABLE_WARP_UTILS_HEADER_

#include "../../../include/common/random.hlsli"
#include "../../../include/common/linear_algebra.hlsli"
#include "../../../include/common/vmf.hlsli"
// #include "../../../raytracer/spt_interface.hlsli"

// float warp_horizon_term(
//     const KernelParameters &kernel_parameters,
//     const Shape *shapes,
//     const ShapeAdjacency *shape_adjacencies,
//     const SurfacePoint &shading_point,
//     const SurfacePoint &aux_point,
//     const Intersection &aux_isect,
//     int rel_vidx
// ) {
//     // Also check for non-manifold edges.
//     auto adj0 = shape_adjacencies[aux_isect.shape_id].adjacency[aux_isect.tri_id * 3 + rel_vidx];
//     auto adj1 = shape_adjacencies[aux_isect.shape_id].adjacency[aux_isect.tri_id * 3 + (rel_vidx + 2) % 3];
//     if (adj0 == -1 || adj1 == -1) {
//         return float(1.0);
//     }

//     auto vidx = get_indices(shapes[aux_isect.shape_id], aux_isect.tri_id)[rel_vidx];
//     std::vector<int> face_indices = shape_adjacencies[aux_isect.shape_id].vertex_adjacency[vidx];

//     Real total_inner_product = Real(0.0);
//     auto w = normalize(aux_point.position - shading_point.position);
//     for (int i = 0; i < face_indices.size(); i++) {
//         int face_idx = face_indices.at(i);
//         auto vidxs = get_indices(shapes[aux_isect.shape_id], face_idx);

//         auto a0 = array_as_vec3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[0]]);
//         auto a1 = array_as_vec3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[1]]);
//         auto a2 = array_as_vec3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[2]]);

//         auto an = normalize(cross(a0 - a1, a1 - a2));

//         total_inner_product += dot(an, w);

//         // If a single adjacent face point away from the source, then
//         // this must be a boundary vertex.
//         if ((dot(an, w) * total_inner_product) < 0) return float(1.0);
//     }

//     float avg_inclination = total_inner_product / face_indices.size();
//     float alpha = 100;
//     float term = alpha / (1.0 - avg_inclination * avg_inclination) - (alpha - 1);
//     return term;
// }

float warp_boundary_term(
    // const KernelParameters &kernel_parameters,
    // const Shape *shapes,
    // const ShapeAdjacency *shape_adjacencies,
    // const Ray &primary,
    // const Ray &auxiliary,
    // const Intersection &aux_isect,
    // const SurfacePoint &aux_point,
    // const SurfacePoint &shading_point,
    // out float horizon_term,
) {
    return 1.f;
    // float boundary_term = 0.0f;
    // if (dot(shading_point.geom_normal, auxiliary.dir) * dot(shading_point.geom_normal, primary.dir) < 1e-4) {
    //     // Outside the horizon ('hit' the black hemispherical occluder)
    //     horizon_term = 0.0;
    //     boundary_term = exp(-abs(dot(shading_point.geom_normal, auxiliary.dir)) / kernel_parameters.asymptoteBoundaryTemp);
    // } else if (aux_isect.valid()) {
    //     // Hit a valid surface (and isn't beyond horizon)
    //     auto vidxs = get_indices(shapes[aux_isect.shape_id], aux_isect.tri_id);

    //     auto p0 = array_as_vec3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[0]]);
    //     auto p1 = array_as_vec3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[1]]);
    //     auto p2 = array_as_vec3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[2]]);
    //     auto p = aux_point.position;

    //     // Compute the edges.
    //     auto e0 = p0 - p1;
    //     auto e1 = p1 - p2;
    //     auto e2 = p2 - p0;

    //     // Compute distance to edges.
    //     auto A = length(cross(e0, e1));
    //     auto w = length(cross(e0, p - p0)) / A; // barycentric of vtx2
    //     auto u = length(cross(e1, p - p1)) / A; // barycentric of vtx0
    //     auto v = length(cross(e2, p - p2)) / A; // barycentric of vtx1

    //     auto horiz0 = warp_horizon_term(kernel_parameters,
    //                                     shapes,
    //                                     shape_adjacencies,
    //                                     shading_point,
    //                                     aux_point,
    //                                     aux_isect, 0);
    //     auto horiz1 = warp_horizon_term(kernel_parameters,
    //                                     shapes,
    //                                     shape_adjacencies,
    //                                     shading_point,
    //                                     aux_point,
    //                                     aux_isect, 1);
    //     auto horiz2 = warp_horizon_term(kernel_parameters,
    //                                     shapes,
    //                                     shape_adjacencies,
    //                                     shading_point,
    //                                     aux_point,
    //                                     aux_isect, 2);

    //     float interpolated_inv_horiz_term = (u / horiz0 + v / horiz1 + w / horiz2) / (u + v + w);
    //     boundary_term = interpolated_inv_horiz_term;
    //     horizon_term = 1.0 / interpolated_inv_horiz_term;
    // } else {
    //     // Hit environment map. No sharp boundaries.
    //     horizon_term = 0.0;
    //     boundary_term = 0.0;
    // }
    // return boundary_term;
}

static const float asymptote_gamma = 2.0;

[Differentiable]
float warp_weight(
    float3 primary_dir,
    no_diff float3 auxiliary_dir,
    no_diff float vMF_concentration,
) {
    const float boundary_term = no_diff warp_boundary_term();
    const float gamma = asymptote_gamma;
    const float k = vMF_concentration / gamma;
    const float gauss = exp(k * (dot(primary_dir, auxiliary_dir) - 1));
    // Compute the harmonic weight.
    const float harmonic = pow(gauss, gamma) / pow(1 - gauss * boundary_term, gamma);
    return harmonic;
}

/*
 * Computes the derivative of the asymptotic weight of the
 * auxiliary ray's contribution to
 * the warp field. This is independent of the parameter.
 */
float3 warp_weight_grad(
    float3 primary_dir,
    float3 auxiliary_dir,
    float vMF_concentration,
) {
    // Compute the boundary term S(x).
    // This is simply inverse exponential distance product.
    // Has no gradient in x. Only dependent on aux.
    // Real horizon_term = 0.0;
    const float boundary_term = warp_boundary_term();
    
    // Compute the inverse gaussian term.
    // auto inv_gauss = exp(square(1 - dot(primary.dir, auxiliary.dir)) / asymptoteInvGaussSigma);
    const float gamma = asymptote_gamma;
    const float k = vMF_concentration / gamma;

    // auto gauss = exp(-square(1 - dot(primary.dir, auxiliary.dir)) / kernel_parameters.asymptoteInvGaussSigma);
    const float gauss = exp(k * (dot(primary_dir, auxiliary_dir) - 1));

    // Gradient of inverse gaussian.
    const float3 inv_gauss_grad = exp(k * (1 - dot(primary_dir, auxiliary_dir))) * k 
        * cross(primary_dir, cross(primary_dir, auxiliary_dir));

    // Compute the harmonic weight.
    const float harmonic = -gamma * pow(gauss, gamma + 1) / pow(1 - boundary_term * gauss, gamma + 1);

    // Compute the gradient of the harmonic weight w.r.t 'wo'.
    const float3 harmonic_gradient = harmonic * inv_gauss_grad;
    // (Note that this is a vector quantity)
    return harmonic_gradient;
}

// [Differentiable]
// float warp_horizon_term(
//     // const KernelParameters &kernel_parameters,
//     // const Shape *shapes,
//     // const ShapeAdjacency *shape_adjacencies,
//     // const SurfacePoint &shading_point,
//     // const SurfacePoint &aux_point,
//     // const Intersection &aux_isect,
//     // int rel_vidx
// ) {
//     // // Also check for non-manifold edges.
//     // auto adj0 = shape_adjacencies[aux_isect.shape_id].adjacency[aux_isect.tri_id * 3 + rel_vidx];
//     // auto adj1 = shape_adjacencies[aux_isect.shape_id].adjacency[aux_isect.tri_id * 3 + (rel_vidx + 2) % 3];
//     // if (adj0 == -1 || adj1 == -1) {
//     //     return Real(1.0);
//     // }

//     // auto vidx = get_indices(shapes[aux_isect.shape_id], aux_isect.tri_id)[rel_vidx];
//     // std::vector<int> face_indices = shape_adjacencies[aux_isect.shape_id].vertex_adjacency[vidx];

//     // Real total_inner_product = Real(0.0);
//     // auto w = normalize(aux_point.position - shading_point.position);
//     // for (int i = 0; i < face_indices.size(); i++) {
//     //     int face_idx = face_indices.at(i);
//     //     auto vidxs = get_indices(shapes[aux_isect.shape_id], face_idx);

//     //     auto a0 = array_as_vec3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[0]]);
//     //     auto a1 = array_as_vec3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[1]]);
//     //     auto a2 = array_as_vec3(&shapes[aux_isect.shape_id].vertices[3 * vidxs[2]]);

//     //     auto an = normalize(cross(a0 - a1, a1 - a2));

//     //     total_inner_product += dot(an, w);

//     //     // If a single adjacent face point away from the source, then
//     //     // this must be a boundary vertex.
//     //     if ((dot(an, w) * total_inner_product) < 0) return Real(1.0);
//     // }

//     // Real avg_inclination = total_inner_product / face_indices.size();
//     // auto alpha = 100;
//     // auto term = alpha / (1.0 - avg_inclination * avg_inclination) - (alpha - 1);

//     // return term;
// }

/**
 * @param grad_xg Gradient w.r.t the intersection point.
 * @param barycentric Barycentric coordinates of the intersection point.
 *                    assuming the bary is (uv) and coord is (1-u-v, u, v).
 */
void splat_isect_gradient(
    float3 grad_xg,
    float2 barycentric,
) {
    // Instead of using d_intersect_shape (which differentiates only 't' directly),
    // we're going about this in a slightly more straightforward manner.
    // This still works because the spatial derivative is distinct from the
    // 'temporal' (w.r.t parameter) derivative.
    // Propagate the xg derivative through to points manually.
    // For this we assume (falsely, but without consequence) that
    // the intersection was originally computed using u * v0 + v * v1 + (1 - (u+v)) * v2 instead of o + r*t.
    // This keeps the original point, but changes it's dependencies to what we actually need.
    float3 d_aux_v_p_0 = (grad_xg) * (1 - (barycentric[0] + barycentric[1]));
    float3 d_aux_v_p_1 = (grad_xg) * barycentric[0];
    float3 d_aux_v_p_2 = (grad_xg) * barycentric[1];

    
    // atomic_add(&d_shapes[t_aux_isect.shape_id].vertices[3 * aux_tri_index[0]],
    //             d_aux_v_p[0]);
    // atomic_add(&d_shapes[t_aux_isect.shape_id].vertices[3 * aux_tri_index[1]],
    //             d_aux_v_p[1]);
    // atomic_add(&d_shapes[t_aux_isect.shape_id].vertices[3 * aux_tri_index[2]],
    //             d_aux_v_p[2]);
}

struct MinimumIsect {
    float3 position;
    bool valid;
    float2 barycentric;
};

bool IsValid(inout_ref(MinimumIsect) isect) {
    return isect.valid;
}

float3x3 warp_jacobian(
    MinimumIsect aux_isect,
    float3 ray_origin,
) {
    // No intersection => no global coordinate derivative.
    if (!IsValid(aux_isect)) {
        return float3x3(0.f);
    }

    // Compute distance vector from current point to the auxiliary intersection.
    const float3 aux_vector = aux_isect.position - ray_origin;
    const float dist = length(aux_vector);
    // Compute the local-global jacobian matrix here.
    return identity3x3() * (1 / dist) -
           outer_product(aux_vector, aux_vector) / (dist * dist * dist);
}

/**
 * @param f: the derivative of the loss w.r.t the pixel.
 */
void was_estimate<let num_aux_rays : int>(
    float3 ray_origin,
    float3 w,
    float kappa,
    float f,
    no_diff inout_ref(RandomSamplerState) RNG,
) {
    vMFDistribution vmf = vMFDistribution(w, kappa);

    float v_aux_weights[num_aux_rays];
    float3 v_aux_ray_dir[num_aux_rays];
    float3 v_aux_div_weights[num_aux_rays];
    MinimumIsect v_aux_isects[num_aux_rays];

    // Algorithm 2: line 3-4
    // Sample w'_i from vMF distribution with mean w
    for (uint i = 0; i < num_aux_rays; i++) {
        // line 3: Sample w'_i from vMF distribution with mean w
        v_aux_ray_dir[i] = vmf.DrawSample(GetNextRandomFloat2(RNG));
        // line 4: Diff-Intersect
        // GeometryHit hit;
        // // Discard degenerate aux rays
        // if (!HasHit(hit)) {

        // }

        // Compute aux weights w(x,x') and their gradient grad_x'(w(x,x'))
        v_aux_weights[i] = warp_weight(w, v_aux_ray_dir[i], kappa);
        v_aux_div_weights[i] = warp_weight_grad(w, v_aux_ray_dir[i], kappa);
    }

    // Algorithm 2: line 5.5
    // Compute PDF of aux-rays.
    // * v_aux_pdfs corresponds to PDF(w'_i)
    float v_aux_pdfs[num_aux_rays];
    for (uint i = 0; i < num_aux_rays; i++) {
        v_aux_pdfs[i] = vmf.Pdf(v_aux_ray_dir[i]);
    }

    // Algorithm 2: line 9-10
    // Compute the normalization terms Z(x) and grad.Z(x) (independent of parameter)
    //  * normalization corresponds to \hat{Z}
    //  * div_normalization corresponds to \hat{\part Z}
    float normalization = 0;
    float3 div_normalization = float3(0, 0, 0);
    for (uint i = 0; i < num_aux_rays; i++) {
        normalization += v_aux_weights[i] / v_aux_pdfs[i];
        div_normalization += v_aux_div_weights[i] / v_aux_pdfs[i];
    }
    // Compute the normalization terms:
    // (i) 'inv_normalization' := reciprocal of the weight integral (normalization) 1/\int_{x'}(w(x, x'))
    // (ii) 'grad_inv_normalization' := derivative of this reciprocal (\int{x'}grad_w(x, x'))/\int_{x'}(w^2(x, x'))
    float inv_normalization[num_aux_rays];
    float3 grad_inv_normalization[num_aux_rays];
    // This is the main source of bias.
    // To handle this we provide two modes simple Monte Carlo (consistent) and Russian roulette
    // (unbiased but higher variance)

    // TODO :: other modes
    // Consistent estimator. Robust and bias low enough for most applications.
    for (int i = 0; i < num_aux_rays; i++) {
        inv_normalization[i] = 1.0 / normalization;
        grad_inv_normalization[i] = div_normalization * (1.0 / normalization) * (1.0 / normalization);
    }
    
    float3 df_dw = float3(0, 0, 0);
    
    for (int i = 0; i < num_aux_rays; i++) {

        if (!IsValid(v_aux_isects[i])) continue;

        // float3x3 dw_ddir; // Jacobian of domain coords w.r.t ray direction
        // float3x3 dw_dorg; // Jacobian of domain coords w.r.t ray origin
        
        // Jacobian of domain coords w.r.t intersection point.
        float3x3 dw_dxg = warp_jacobian(v_aux_isects[i], ray_origin);

        // Algorithm 2: line 12
        // Compute warp field contribution from this auxiliary ray.
        // 3x3 (3D vector field in domain, 3 spatial parameters)
        float vMultiplier = (v_aux_weights[i] / v_aux_pdfs[i]) * inv_normalization[i];
        // Algorithm 2: line 8: apply the jacobbian to the vector field.
        float3x3 V_xg = dw_dxg * vMultiplier;
        
        // Algorithm 2: line 13
        // Compute contribution to the divergence of warp field.
        // 3x1 (scalar field in domain, 3 spatial parameters)
        float3 divVMultiplier = (v_aux_div_weights[i] / v_aux_pdfs[i]) * inv_normalization[i] -
                                (v_aux_weights[i] / v_aux_pdfs[i]) * grad_inv_normalization[i];
        // Algorithm 2: line 8: apply the jacobbian to the vector field.
        float3 divV_xg = mul(divVMultiplier, dw_dxg);

        // Algorithm 1: line 6
        // Compute the gradient of the loss I w.r.t the intersection point (world space).
        // It consists of two terms:
        float3 gradF_dot_V_xg = mul(df_dw, V_xg); // lets omot screen-space kernel first: + f_gradK * V_xg;
        float3 F_mul_div_V_xg = f * divV_xg;  // F.K.divV
        // Then add up to compute the final gradients.
        float3 grad_xg = gradF_dot_V_xg + F_mul_div_V_xg;
        
        // Gradient w.r.t ray direction.
        // TODO:: Check hit.valid() here.
        float2 barycentric = float2(0, 0); // TODO :: get the barycentric coordinates
        splat_isect_gradient(grad_xg, barycentric);
    }
}

#endif // _SRENDERER_DIFFERENTIABLE_WARP_UTILS_HEADER_