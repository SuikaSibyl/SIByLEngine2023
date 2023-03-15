#ifndef _SRENDERER_COMMON_MICROFACET_HEADER_
#define _SRENDERER_COMMON_MICROFACET_HEADER_

#include "../../Utility/math.h"
#include "../../Utility/geometry.h"
#include "spectrum.h"

/**
* The Schlick Fresnel approximation is:
*   R = R(0) + (1 - R(0)) (1 - cos theta)^5
* where R(0) is the reflectance at normal indicence.
*/

float SchlickWeight(in float cosTheta) {
    const float m = clamp(1.0 - cosTheta, 0.0, 1.0);
    const float m2 = m*m;
    return m2*m2*m; // pow(m,5)
}

float GTR2_aniso(float NDotH, float HDotX, float HDotY, float ax, float ay) {
    float a = HDotX / ax;
    float b = HDotY / ay;
    float c = a * a + b * b + NDotH * NDotH;
    return 1.0 / (k_pi * ax * ay * c * c);
}

/// Fresnel equation of a dielectric interface.
/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
/// n_dot_i: abs(cos(incident angle))
/// n_dot_t: abs(cos(transmission angle))
/// eta: eta_transmission / eta_incident
float fresnel_dielectric(in const float n_dot_i, in const float n_dot_t, in const float eta) {
    float rs = (n_dot_i - eta * n_dot_t) / (n_dot_i + eta * n_dot_t);
    float rp = (eta * n_dot_i - n_dot_t) / (eta * n_dot_i + n_dot_t);
    float F = (rs * rs + rp * rp) / 2;
    return F;
}

/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
/// This is a specialized version for the code above, only using the incident angle.
/// The transmission angle is derived from 
/// n_dot_i: cos(incident angle) (can be negative)
/// eta: eta_transmission / eta_incident
float fresnel_dielectric(in const float n_dot_i, in const float eta) {
    float n_dot_t_sq = 1 - (1 - n_dot_i * n_dot_i) / (eta * eta);
    if (n_dot_t_sq < 0) {
        // total internal reflection
        return 1;
    }
    float n_dot_t = sqrt(n_dot_t_sq);
    return fresnel_dielectric(abs(n_dot_i), n_dot_t, eta);
}

// Anisotropic GGX
float GGX_aniso(in const float ax, in const float ay, in const vec3 H, in const mat3 frame) {
    const vec3 hl = to_local(frame, H);
    const float d = hl.x * hl.x / (ax * ax) + hl.y * hl.y / (ay * ay) + hl.z * hl.z;
    return float(1) / (k_pi * ax * ay * d * d);
}

float smith_masking_gtr2_aniso(in const vec3 v_local, in const float ax, in const float ay) {
    const float term1 = v_local.x * ax;
    const float term2 = v_local.y * ay;
    const float Lambda = (-1 + sqrt(1 + ((term1 * term1 + term2 * term2) / (v_local.z * v_local.z)))) / 2;
    return float(1) / (1 + Lambda);
}

float GTR2(in const float n_dot_h, in const float roughness) {
    const float alpha = roughness * roughness;
    const float a2 = alpha * alpha;
    const float t = 1 + (a2 - 1) * n_dot_h * n_dot_h;
    return a2 / (k_pi * t*t);
}

float smith_masking_gtr2(in const vec3 v_local, in const float roughness) {
    const float alpha = roughness * roughness;
    const float a2 = alpha * alpha;
    const vec3  v2 = v_local * v_local;
    const float Lambda = (-1 + sqrt(1 + (v2.x * a2 + v2.y * a2) / v2.z)) / 2;
    return 1 / (1 + Lambda);
}

/// See "Sampling the GGX Distribution of Visible Normals", Heitz, 2018.
/// https://jcgt.org/published/0007/04/01/
vec3 sample_visible_normals(
    in vec3  local_dir_in,
    in const float alpha, 
    in const vec2  rnd_param
) {
    // The incoming direction is in the "ellipsodial configuration" in Heitz's paper
    float negative = 1;
    if (local_dir_in.z < 0) {
        // Ensure the input is on top of the surface.
        local_dir_in = -local_dir_in;
        negative = -1;
    }

    // Transform the incoming direction to the "hemisphere configuration".
    vec3 hemi_dir_in = normalize(vec3(alpha * local_dir_in.x, alpha * local_dir_in.y, local_dir_in.z));

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
    vec3 disk_N = vec3(t1, t2, sqrt(max(0, 1 - t1*t1 - t2*t2)));

    // Reprojection onto hemisphere -- we get our sampled normal in hemisphere space.
    mat3 hemi_frame = createFrame(hemi_dir_in);
    vec3 hemi_N = to_world(hemi_frame, disk_N);

    // Transforming the normal back to the ellipsoid configuration
    return negative * normalize(vec3(alpha * hemi_N.x, alpha * hemi_N.y, max(0, hemi_N.z)));
}

#endif