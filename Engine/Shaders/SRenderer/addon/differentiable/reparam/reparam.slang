#include "../../../include/common/light_impl.hlsli"
#include "../../../include/common/sampling.hlsli"
#include "../../../include/common/vmf.hlsli"
#include "../../../include/common/rotation.hlsli"
#include "../../../include/common/raydifferential.hlsli"
#include "../../../include/diff_descriptor_set.hlsli"
#include "../../../raytracer/spt_interface.hlsli"
#include "../../../raytracer/spt_differentiable.hlsli"
#include "reparam.hlsli"

// Changes of variables in pixel integrals using %i samples
#define m_dc_cam_samples 4

[Differentiable]
float3 reparameterizing_disconitnuous_integrands(
    in_ref(Ray) prim_ray,
    in_ref(RayDifferential) prim_raydiff,
    inout_ref(RandomSamplerState) RNG,
) {
    float3 radiance = float3(0);
    PrimaryPayload payload;

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

    float3 dir_conv_0;
    float3 dir_conv_1;
    if (true) {
        // ---------------- Convolution of pixel integrals ----------------
        // Detect discontinuities in a small vMF kernel around each ray.
        Ray rays[m_dc_cam_samples];
        GeometryHit sis[m_dc_cam_samples];
        // Sample the integrals and gather intersections
        vMFDistribution vmf = vMFDistribution(prim_ray.direction, kappa_camera);
        for (int cs = 0; cs < int(m_dc_cam_samples); cs++) {
            Ray primary_ray = prim_ray;
            primary_ray.direction = vmf.DrawSample(GetNextRandomFloat2(RNG));

            Intersection(primary_ray, SceneBVH, payload, RNG);
            sis[cs] = payload.hit;
            // sis[cs].compute_differentiable_shape_position(active_primary);

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

        // The following rotation seems to be the identity transformation,
        // but it actually changes the partial derivatives.

        // Create the differentiable rotation
        float3 axis = cross(detach(discontinuity_dir), discontinuity_dir);
        float cosangle = dot(discontinuity_dir, detach(discontinuity_dir));
        float4x4 rotation = rotation_from_axis_cosangle(axis, cosangle);

        // Tracks radiance scaling due to index of refraction changes
        float eta = (1.f);

        throughput *= vMF_pdf_diff / detach(vMF_pdf_diff); // NOTE: detach here so we only divide the gradient by the pdf

        // for (uint depth = 1; ; ++depth) {
        if (true) {
            // ---------------- Intersection with emitters ----------------
            float3 emission_0 = float3(0.f);
            float3 emission_1 = float3(0.f);

            emission[active] = throughput * emitter->eval(si, active);

            float weights_0;
            float weights_1;

            radiance += emission_0 * 0.5f;
            radiance += emission_1 * 0.5f;
        }
        
        float3 dir_conv = discontinuity_dir;
    }

    return radiance;
}