#ifndef _SRENDERER_COMMON_MICROFACET_HEADER_
#define _SRENDERER_COMMON_MICROFACET_HEADER_

#include "../../Utility/math.h"

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

float luminance(in const vec3 s) {
    return s.x * float(0.212671) + s.y * float(0.715160) + s.z * float(0.072169);
}

#endif