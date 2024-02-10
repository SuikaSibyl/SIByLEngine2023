#include "../common/random.hlsli"
#include "../common/rotation.hlsli"
#include "../common/vmf.hlsli"
#include "common.hlsli"
#define m_dc_cam_samples 4

[[vk::push_constant]]
cbuffer PushConstants { int rand_seed; };

RWStructuredBuffer<float> u_gradient;
RWStructuredBuffer<float3> u_trans;
RWByteAddressBuffer u_trans_gradient;

/**
 * @generic nb_samples: number of samples, must be larger than 2
 */
[Differentiable]
float3 estimate_discontinuity<let nb_samples : int>(
    in_ref(Ray) rays[nb_samples],
    in_ref(HitAD) sis[nb_samples],
) {
    float3 ray0_p_attached = sis[0].position;
    float3 ray0_n = sis[0].normal;

    // For all the other samples (indices are not 0),
    // try to find one has different geometryID
    uint is_ray1_hit_uint = select(HasHit(sis[1]), 1u, 0u);
    float3 ray1_p_attached = sis[1].position;
    float3 ray1_n = sis[1].normal;
    float3 ray1_d = rays[1].direction;

    [ForceUnroll]
    for (uint i = 2; i < nb_samples; i++) {
        const bool diff = (sis[0].geometryID != sis[i].geometryID);
        const bool i_hit = HasHit(sis[i]);
        is_ray1_hit_uint = select(diff, select(i_hit, 1u, 0u), is_ray1_hit_uint);
        ray1_p_attached = select(diff, sis[i].position, ray1_p_attached);
        ray1_n = select(diff, sis[i].normal, ray1_n);
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

[Differentiable]
float3 reparameterizing_disconitnuous_integrands(
    no_diff Ray prim_ray,
    no_diff RayDifferential prim_raydiff,
    no_diff inout_ref(RandomSamplerState) RNG,
    float3 translation
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

    float3 radiance = float3(0);
    float3 dir_conv_0;
    float3 dir_conv_1;
    // ---------------- Convolution of pixel integrals ----------------
    // Detect discontinuities in a small vMF kernel around each ray.
    Ray rays[m_dc_cam_samples];
    HitAD sis[m_dc_cam_samples];
    // Sample the integrals and gather intersections
    vMFDistribution vmf = vMFDistribution(prim_ray.direction, kappa_camera);
    [ForceUnroll]
    for (int cs = 0; cs < int(m_dc_cam_samples); cs++) {
        Ray primary_ray = prim_ray;
        primary_ray.direction = no_diff vmf.DrawSample(GetNextRandomFloat2(RNG));

        // Intersection(primary_ray, SceneBVH, payload, RNG);
        HitAD hit = no_diff ray_tracing(primary_ray, translation);
        sis[cs] = hit;
        loadPosition(sis[cs], translation);

        rays[cs] = primary_ray;

        // Keep two directions for creating pairs of paths.
        // We choose the last samples since they have less
        // chances of being used in the estimation of the
        // discontinuity.
        if (cs == m_dc_cam_samples - 2)
            dir_conv_0 = primary_ray.direction;
        if (cs == m_dc_cam_samples - 1)
            dir_conv_1 = primary_ray.direction;
    }

    const float3 discontinuity = estimate_discontinuity(rays, sis);
    const float3 discontinuity_dir = normalize(discontinuity - prim_ray.origin);

    // The following rotation seems to be the identity transformation, but it actually
    // changes the partial derivatives.

    // Create the differentiable rotation
    float3 axis = cross(detach(discontinuity_dir), discontinuity_dir);
    float cosangle = dot(discontinuity_dir, detach(discontinuity_dir));
    float3x3 rotation = rotation_from_axis_cosangle(axis, cosangle);

    // NOTE: here we detach because the rays will be passed to Optix, no need for autodiff
    float3 ray_d_0 = mul(rotation, detach(dir_conv_0));
    float3 ray_d_1 = mul(rotation, detach(dir_conv_1));

    Ray ray_0; ray_0.origin = prim_ray.origin; ray_0.direction = ray_d_0;
    Ray ray_1; ray_1.origin = prim_ray.origin; ray_1.direction = ray_d_1;

    // Recompute differentiable pdf
    float vMF_pdf_diff_0 = vmf.Pdf(ray_d_0);
    float vMF_pdf_diff_1 = vmf.Pdf(ray_d_1);

    // Apply differentiable weight and keep for variance reduction
    // NOTE: detach here so we only divide the gradient by the pdf
    float3 throughput_0 = float3(1.0) * vMF_pdf_diff_0 / detach(vMF_pdf_diff_0);
    float3 throughput_1 = float3(1.0) * vMF_pdf_diff_1 / detach(vMF_pdf_diff_1);

    HitAD hit_0 = no_diff ray_tracing(ray_0, translation);
    HitAD hit_1 = no_diff ray_tracing(ray_1, translation);

    float3 emission_0 = throughput_0 * hit_0.color;
    float3 emission_1 = throughput_1 * hit_1.color;

    bool use_variance_reduction = true;
    if (use_variance_reduction) {
        float weights_0 = vMF_pdf_diff_0 / detach(vMF_pdf_diff_0);
        float weights_1 = vMF_pdf_diff_1 / detach(vMF_pdf_diff_1);

        // Avoid numerical errors due to tiny weights
        weights_0 = select(abs(weights_0) < 0.00001f, (1.f), weights_0);
        weights_1 = select(abs(weights_1) < 0.00001f, (1.f), weights_1);

        // Variance reduction, assumption that contribution = weight * constant
        radiance += (emission_0 - emission_1 / weights_1 * (weights_0 - detach(weights_0))) * 0.5f; // NOTE: detach here so to only add `e_1/w_1*w_0` to the gradient (only try to reduce the variance of the gradient)
        radiance += (emission_1 - emission_0 / weights_0 * (weights_1 - detach(weights_1))) * 0.5f;
    } else {
        radiance += emission_0 * 0.5f;
        radiance += emission_1 * 0.5f;
    }

    return radiance;
}

[Differentiable]
float forward_render(
    no_diff Ray prim_ray,
    no_diff RayDifferential prim_raydiff,
    no_diff inout_ref(RandomSamplerState) RNG,
    float3 translation = float3(0),
) {
    float3 radiance = reparameterizing_disconitnuous_integrands(prim_ray, prim_raydiff, RNG, translation);
    return radiance.x;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void ComputeMain(int3 dtid: SV_DispatchThreadID) {
    // The resolution of the image, which is the same as the launch size:
    const int2 resolution = int2(512, 512);
    const int2 pixel = int2(dtid.xy);

    // If the pixel is outside of the image, don't do anything:
    if (any(pixel >= resolution)) return;

    RandomSamplerState RNG = InitRandomSampler(pixel, rand_seed);
    Ray ray = generateRay(pixel, resolution);
    RayDifferential raydiff = generateRayDifferential(pixel, resolution);

    const float3 translation = u_trans[0];  // gt: float3(0.1, -0.1, 0.1)
    var trans_pair = diffPair(translation); // constructs DifferentialPair{2.0, 0.0}
    float img_grad = u_gradient[pixel.x * 512 + pixel.y];
    bwd_diff(forward_render)(ray, raydiff, RNG, trans_pair, img_grad);
    
    // u_gradient[pixel.x * 512 + pixel.y] = forward_render(ray, raydiff, RNG, translation) + rand_seed;
    // u_gradient[pixel.x * 512 + pixel.y] = ray_tracing(ray, translation).color.x + rand_seed;

    if (!any(isnan(trans_pair.d) || isinf(trans_pair.d))) {
        u_trans_gradient.InterlockedAddF32(0, trans_pair.d.x);
        u_trans_gradient.InterlockedAddF32(4, trans_pair.d.y);
        u_trans_gradient.InterlockedAddF32(8, trans_pair.d.z);
        // u_trans_gradient.InterlockedAddF32(0, 1);
        // u_trans_gradient.InterlockedAddF32(4, 1);
        // u_trans_gradient.InterlockedAddF32(8, 1);
    }
}