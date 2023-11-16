RWTexture2D<float4> u_color;
RWTexture2D<float4> u_loss;
RWByteAddressBuffer u_buffer;

[[vk::push_constant]]
cbuffer PushConstants {
    int rand_seed;
};

// Changes of variables in pixel integrals using %i samples
#include "../../../include/common/colormaps.hlsli"
#include "../../../include/common/random.hlsli"
#include "../../../include/common/rotation.hlsli"
#include "../../../include/common/vmf.hlsli"
#include "../common/simple-test-utils.hlsli"
#include "was-utils.hlsli"

/**
 * @param num_aux_rays: number of auxiliary rays to use for WAS.
 * @param ray_origin: the origin of the shading point now.
 * @param w: the central direction of the ray.
 * @param kappa: the concentration parameter of the vMF distribution for auxiliary rays.
 * @param f: the derivative of the loss w.r.t the hitting point.
 * @param RNG: the random number generator.
 * @param translation: the translation of the triangle in the simple scene.
 */
float4 was_estimate_simple<let num_aux_rays : int>(
    float3 ray_origin,
    float3 w,
    float kappa,
    float f,
    no_diff inout_ref(RandomSamplerState) RNG,
    float3 translation,
) {
    vMFDistribution vmf = vMFDistribution(w, kappa);
    float4 debug = float4(0, 0, 0, 1);
    float4 debug2 = float4(0, 0, 0, 1);

    float v_aux_weights[num_aux_rays];
    float3 v_aux_ray_dir[num_aux_rays];
    float3 v_aux_div_weights[num_aux_rays];
    MinimumIsect v_aux_isects[num_aux_rays];
    float v_aux_pdfs[num_aux_rays];
    
    // Algorithm 2: line 3-4
    // Sample w'_i from vMF distribution with mean w
    for (uint i = 0; i < num_aux_rays; i++) {
        // line 3: Sample w'_i from vMF distribution with mean w
        v_aux_ray_dir[i] = vmf.DrawSample(GetNextRandomFloat2(RNG));
        // line 4: Diff-Intersect
        // We do not really need to have this process differentiable here.
        // Which means we simply get y, and handle all derivatives later.
        Ray aux_ray = Ray(ray_origin, v_aux_ray_dir[i]);
        HitAD aux_hit = ray_tracing(aux_ray, float3(0.1, -0.1, 0.1));
        // Discard degenerate aux rays
        if (!HasHit(aux_hit)) {
            v_aux_isects[i].valid = false;
        } else {
            v_aux_isects[i].valid = true;
            v_aux_isects[i].position = aux_hit.position;
            v_aux_isects[i].barycentric = aux_hit.uv;
        }

        // Algorithm 2: line 5.5
        // Compute PDF of aux-rays.
        // * v_aux_pdfs corresponds to PDF(w'_i)
        v_aux_pdfs[i] = vmf.Pdf(v_aux_ray_dir[i]);

        // Compute aux weights w(x,x') and their gradient grad_x'(w(x,x'))
        var w_pair = diffPair(w, 1.0);
        var ray_dir_pair = fwd_diff(warp_weight)(w_pair, v_aux_ray_dir[i], kappa);
        // var ray_dir_conv = warp_weight(w, v_aux_ray_dir[i], kappa);
        v_aux_weights[i] = ray_dir_pair.p;
        debug2 = float4(ray_dir_pair.p, 0, 0, 1);
        v_aux_div_weights[i] = warp_weight_grad(w, v_aux_ray_dir[i], kappa);
        // v_aux_weights[i] = warp_weight(w, v_aux_ray_dir[i], kappa);
        // v_aux_div_weights[i] = warp_weight_grad(w, v_aux_ray_dir[i], kappa);
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
        float3 F_mul_div_V_xg = f * divV_xg;      // F.K.divV
        // Then add up to compute the final gradients.
        float3 grad_xg = gradF_dot_V_xg + F_mul_div_V_xg;

        // Gradient w.r.t ray direction.
        if (v_aux_isects[i].valid == false) continue;
        // debug += float4(float3(v_aux_div_weights[i]), 0);
        
        // TODO :: Go back and fix this later
        if(any(isnan(grad_xg) || isinf(grad_xg))) continue;
        float2 barycentric = v_aux_isects[i].barycentric;
        // splat_isect_gradient(grad_xg, barycentric);
        var translation_pair = diffPair(translation);
        bwd_diff(loadPosition)(v_aux_isects[i].barycentric, translation_pair, grad_xg);
        
        debug += float4(translation_pair.d, 0.0);
    }

    // accumulate the gradient of the loss w.r.t the ray direction.
    if (!any(isnan(debug.xyz) || isinf(debug.xyz))) {
        u_buffer.InterlockedAddF32(16, debug.xyz.x);
        u_buffer.InterlockedAddF32(20, debug.xyz.y);
        u_buffer.InterlockedAddF32(24, debug.xyz.z);
    }
    
    return debug2;
}

float4 primal_was_estimate(
    Ray prim_ray,
    RayDifferential prim_raydiff,
    float3 didpx,
    inout_ref(RandomSamplerState) RNG,
    float3 translation,
) {
    // Estimate kappa for the convolution of pixel integrals,
    // based on ray differentials.
    const float angle = acos(min(
        dot(prim_raydiff.dddx, prim_ray.direction),
        dot(prim_raydiff.dddy, prim_ray.direction)));
    const float target_mean_cos =
        min(cos(angle * 0.4f /*arbitrary*/), float(1.f - 1e-7f));
    // The vMF distribution has an analytic expression for the mean cosine:
    //      mean = 1 + 2/(exp(2*k)-1) - 1/k.
    // For large values of kappa, 1-1/k is a precise approximation of this
    // function. It can be inverted to find k from the mean cosine.
    const float kappa_camera = 1.f / (1.f - target_mean_cos);
    float4 debug =
        was_estimate_simple<1>(prim_ray.origin, prim_ray.direction, kappa_camera, dot(didpx, float3(1)), RNG, translation);
    return debug;
}

[Differentiable]
float L2(
    no_diff Ray prim_ray,
    no_diff RayDifferential prim_raydiff,
    float3 radiance,
) {
    HitAD reference = no_diff ray_tracing(prim_ray, float3(0.1, -0.1, 0.1));
    float3 ref_color = no_diff detach(reference.color);
    return dot((radiance - ref_color), (radiance - ref_color));
}

[shader("raygeneration")]
void RgenMain() {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(DispatchRaysDimensions().xy);
    const int2 pixel = int2(DispatchRaysIndex().xy);
    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;
    RandomSamplerState RNG = InitRandomSampler(pixel, rand_seed);

    Ray ray = generateRay(pixel, resolution);
    RayDifferential raydiff = generateRayDifferential(pixel, resolution);

    
    // // u_color[pixel] = float4(reference.position, 1.0);
    // // // float3 reparameterizing_disconitnuous_integrands(ray, raydiff);

    // (x-a)^2
    // 2x
    float3 translation = u_buffer.Load<float3>(0);
    
    float3 radiance = ray_tracing(ray, translation).color;
    float l2 = L2(ray, raydiff, radiance);
    var radiance_pair = diffPair(radiance, float3(1.0)); // constructs DifferentialPair{2.0, 0.0}
    bwd_diff(L2)(ray, raydiff, radiance_pair, 1.0);

    u_loss[pixel] = float4(float3(l2), 1.0);

    u_color[pixel] = primal_was_estimate(ray, raydiff, radiance_pair.d, RNG, translation);
}