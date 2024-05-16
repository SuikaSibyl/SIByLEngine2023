#ifndef _SRENDERER_COMMON_MICROFACET_HEADER_
#define _SRENDERER_COMMON_MICROFACET_HEADER_

#include "cpp_compatible.hlsli"
#include "math.hlsli"
#include "geometry.hlsli"

/**
 * The Lambertian BRDF is:
 *   f_r = rho / pi
 * where rho is the albedo.
 * @param albedo The albedo of the surface.
 * @return The BRDF value, a.k.a., rho / pi
 */
float3 Lambert(in_ref(float3) albedo) {
    return albedo * k_inv_pi;
}

/**
 * The Schlick Fresnel approximation is:
 *   R = R(0) + (1 - R(0)) (1 - cos theta)^5
 * where R(0) is the reflectance at normal indicence.
 * @param VdotH The cosine of the angle between the half vector and view vector.
 * @return The schlick weight, a.k.a., (1 - (h*v))^5
 */
float SchlickWeight(float VdotH) {
    const float m = clamp(1.0 - VdotH, 0.0, 1.0);
    const float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

/**
 * The Schlick Fresnel approximation is:
 *   R = R(0) + (1 - R(0)) (1 - cos theta)^5
 * where R(0) is the reflectance at normal indicence.
 * @param VdotH The cosine of the angle between the half vector and view vector.
 * @return The Schlick Fresnel approximation result.
 */
float SchlickFresnel(float F0, float VdotH) {
    return F0 + (1 - F0) * SchlickWeight(VdotH); }
float2 SchlickFresnel(in_ref(float2) F0, float VdotH) {
    return F0 + (1 - F0) * SchlickWeight(VdotH); }
float3 SchlickFresnel(in_ref(float3) F0, float VdotH) {
    return F0 + (1 - F0) * SchlickWeight(VdotH); }
float4 SchlickFresnel(in_ref(float4) F0, float VdotH) {
    return F0 + (1 - F0) * SchlickWeight(VdotH); }

/**
 * Generalized Trowbridge and Reitz (GTR) Normal Distribution Function.
 * @url: https://www.disneyanimation.com/publications/physically-based-shading-at-disney
 * @param NdotH The cosine of the angle between the half vector and normal vector.
 * @param roughness The roughness of the surface.
 */
[Differentiable]
float GTR2_NDF(float n_dot_h, float roughness) {
    const float alpha = roughness * roughness;
    const float a2 = alpha * alpha;
    const float t = 1 + (a2 - 1) * n_dot_h * n_dot_h;
    return a2 / (k_pi * t * t);
}

/**
 * Anisotropic Generalized Trowbridge and Reitz (GTR) Normal Distribution Function.
 * @url: https://www.disneyanimation.com/publications/physically-based-shading-at-disney
 * @param NdotH The cosine of the angle between the half vector and normal vector.
 * @param HDotX The cosine of the angle between the half vector and tangent vector.
 * @param HDotY The cosine of the angle between the half vector and bitangent vector.
 * @param ax The roughness of the surface in x direction.
 * @param ay The roughness of the surface in y direction.
 */
float AnisotropicGTR2_NDF(float NDotH, float HDotX, float HDotY, float ax, float ay) {
    const float a = HDotX / ax;
    const float b = HDotY / ay;
    const float c = a * a + b * b + NDotH * NDotH;
    return 1.0 / (k_pi * ax * ay * c * c);
}

/**
 * GGX Normal Distribution Function.
 * Which is equivalent to the Generalized Trowbridge and Reitz (GTR) Normal Distribution Function.
 * @param NdotH The cosine of the angle between the half vector and normal vector.
 * @param roughness The roughness of the surface.
 */
float GGX_NDF(float n_dot_h, float roughness) {
    return GTR2_NDF(n_dot_h, roughness);
}

/**
 * Anisotropic GGX Normal Distribution Function.
 * Which is equivalent to the Generalized Trowbridge and Reitz (GTR) Normal Distribution Function.
 * @param H The half vector in world space.
 * @param frame The shading frame.
 * @param ax The roughness of the surface in x direction.
 * @param ay The roughness of the surface in y direction.
 */
float AnisotropicGGX_NDF(in_ref(float3) H, in_ref(float3x3) frame, float ax, float ay) {
    const float3 hl = to_local(frame, H);
    return AnisotropicGTR2_NDF(hl.z, hl.x, hl.y, ax, ay);
}

/**
 * Isotropic GGX Masking Term.
 * @url: https://jcgt.org/published/0003/02/03/paper.pdf
 * @param v_local The vector in local shading frame.
 * @param roughness The roughness of the surface.
 */
[Differentiable]
float IsotropicGGX_Masking(no_diff float3 v_local, float roughness) {
    const float alpha = roughness * roughness;
    const float a2 = alpha * alpha;
    const float3 v2 = v_local * v_local;
    const float Lambda = (-1 + sqrt(1 + (v2.x * a2 + v2.y * a2) / v2.z)) / 2;
    return 1 / (1 + Lambda);
}

/**
 * Anisotropic GGX Masking Term.
 * @url: https://jcgt.org/published/0003/02/03/paper.pdf
 * @param v_local The vector in local shading frame.
 * @param ax The roughness of the surface in x direction.
 * @param ay The roughness of the surface in y direction.
 */
float AnisotropicGGX_Masking(in_ref(float3) v_local, float ax, float ay) {
    const float ax2 = ax * ax;
    const float ay2 = ay * ay;
    const float3 v2 = v_local * v_local;
    const float Lambda = (-1 + sqrt(1 + (v2.x * ax2 + v2.y * ay2) / v2.z)) / 2;
    return float(1) / (1 + Lambda);
}

/**
 * Fresnel equation of a dielectric interface.
 * (Not Schlick Fresnel approximation)
 * @url: https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
 * @param n_dot_i: abs(cos(incident angle))
 * @param n_dot_t: abs(cos(transmission angle))
 * @param eta: eta_transmission / eta_incident
 */
[Differentiable]
float FresnelDielectric(float n_dot_i, float n_dot_t, float eta) {
    const float rs = (n_dot_i - eta * n_dot_t) / (n_dot_i + eta * n_dot_t);
    const float rp = (eta * n_dot_i - n_dot_t) / (eta * n_dot_i + n_dot_t);
    const float F = (rs * rs + rp * rp) / 2;
    return F;
}

/**
 * Fresnel equation of a dielectric interface.
 * This is a specialized version for the code above, only using the incident angle.
 * The transmission angle is derived from n_dot_i
 * (Not Schlick Fresnel approximation)
 * @url: https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
 * @param n_dot_i: abs(cos(incident angle))
 * @param eta: eta_transmission / eta_incident
 */
[Differentiable]
float FresnelDielectric(float n_dot_i, float eta) {
    const float n_dot_t_sq = 1 - (1 - n_dot_i * n_dot_i) / (eta * eta);
    if (n_dot_t_sq < 0) {
        // total internal reflection
        return 1.f;
    }
    float n_dot_t = sqrt(n_dot_t_sq);
    return FresnelDielectric(abs(n_dot_i), n_dot_t, eta);
}

/**
 * Sample the GGX distribution of visible normals.
 * See "Sampling the GGX Distribution of Visible Normals", Heitz, 2018.
 * @url: https://jcgt.org/published/0007/04/01/
 */
float3 SampleVisibleNormals(
    in_ref(float3) local_dir_in,
    float alpha,
    in_ref(float2) rnd_param
) {
    // The incoming direction is in the "ellipsodial configuration" in Heitz's paper
    float negative = 1;
    if (local_dir_in.z < 0) {
        // Ensure the input is on top of the surface.
        local_dir_in = -local_dir_in;
        negative = -1;
    }

    // Transform the incoming direction to the "hemisphere configuration".
    float3 hemi_dir_in = normalize(float3(alpha * local_dir_in.x, alpha * local_dir_in.y, local_dir_in.z));

    // Parameterization of the projected area of a hemisphere.
    // First, sample a disk.
    float r = sqrt(rnd_param.x);
    float phi = 2 * k_pi * rnd_param.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    // Vertically scale the position of a sample to account for the projection.
    float s = (1 + hemi_dir_in.z) / 2;
    t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
    // Point in the disk space
    float3 disk_N = float3(t1, t2, sqrt(max(0, 1 - t1 * t1 - t2 * t2)));

    // Reprojection onto hemisphere -- we get our sampled normal in hemisphere space.
    float3x3 hemi_frame = createFrame(hemi_dir_in);
    float3 hemi_N = to_world(hemi_frame, disk_N);

    // Transforming the normal back to the ellipsoid configuration
    return negative * normalize(float3(alpha * hemi_N.x, alpha * hemi_N.y, max(0, hemi_N.z)));
}

#endif // !_SRENDERER_COMMON_MICROFACET_HEADER_