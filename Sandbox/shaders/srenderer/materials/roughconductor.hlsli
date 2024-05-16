#ifndef _SRENDERER_ROUGHCONDUCTOR_MATERIAL_
#define _SRENDERER_ROUGHCONDUCTOR_MATERIAL_

#include "../../common/math.hlsli"
#include "../../common/microfacet.hlsli"
#include "../spt.hlsli"

struct RoughConductorMaterial : IDifferentiable {
    float3 albedo;
    float eta; // Relative refractive index (real component).
    float k;   // Relative refractive index (imaginary component).
};

[Differentiable]
float safe_sqrt(float a) {
    return sqrt(max(a, 0));
}

[Differentiable]
float fresnel_conductor(float cos_theta_i, float2 eta) {
    // Modified from "Optics" by K.D. Moeller, University Science Books, 1988
    float cos_theta_i_2 = cos_theta_i * cos_theta_i;
    float sin_theta_i_2 = 1.f - cos_theta_i_2;
    float sin_theta_i_4 = sin_theta_i_2 * sin_theta_i_2;
    float eta_r = eta.x;
    float eta_i = eta.y;
    float temp_1 = eta_r * eta_r - eta_i * eta_i - sin_theta_i_2;
    float a_2_pb_2 = safe_sqrt(temp_1 * temp_1 + 4.f * eta_i * eta_i * eta_r * eta_r);
    float a = safe_sqrt(.5f * (a_2_pb_2 + temp_1));
    float term_1 = a_2_pb_2 + cos_theta_i_2;
    float term_2 = 2.f * cos_theta_i * a;
    float r_s = (term_1 - term_2) / (term_1 + term_2);
    float term_3 = a_2_pb_2 * cos_theta_i_2 + sin_theta_i_4;
    float term_4 = term_2 * sin_theta_i_2;
    float r_p = r_s * (term_3 - term_4) / (term_3 + term_4);
    return .5f * (r_s + r_p);
}

[Differentiable]
float3 EvalRoughConductor(
    RoughConductorMaterial material,
    no_diff BSDFEvalGeometry evalGeom
) {
    if (dot(evalGeom.geometric_normal, evalGeom.dir_in) < 0 ||
        dot(evalGeom.geometric_normal, evalGeom.dir_out) < 0) {
        // No light below the surface
        return float3(0);
    }

    // Flip the shading frame if it is inconsistent with the geometry normal
    float3x3 frame = evalGeom.frame;

    // Calculate the half-direction vector
    float3 H = normalize(evalGeom.dir_out + evalGeom.dir_in);
    // Evaluate the Fresnel factor
    float F = fresnel_conductor(dot(evalGeom.dir_in, H), float2(material.eta, material.k));

    /* Evaluate Smith's shadow-masking function */
    float G_in = IsotropicGGX_Masking(to_local(frame, evalGeom.dir_in), roughness);

    /* Evaluate the full microfacet model (except Fresnel) */
    Value result = D * G / (4.f * Frame::cos_theta(si.wi));

    return F * result * active;
}

#endif // _SRENDERER_ROUGHCONDUCTOR_MATERIAL_