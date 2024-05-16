#ifndef _SRENDERER_ROUGHPLASTIC_HEADER_
#define _SRENDERER_ROUGHPLASTIC_HEADER_

#include "../../common/math.hlsli"
#include "../../common/microfacet.hlsli"
#include "../spt.hlsli"

struct RoughPlasticMaterial : IDifferentiable {
    float3 Kd;
    float eta;
    float3 Ks;
    float roughness;
};

[Differentiable]
float3 EvalRoughPlastic(
    RoughPlasticMaterial material,
    no_diff BSDFEvalGeometry evalGeom
) {
    if (dot(evalGeom.geometric_normal, evalGeom.dir_in) < 0 ||
        dot(evalGeom.geometric_normal, evalGeom.dir_out) < 0) {
        // No light below the surface
        return float3(0);
    }
    // Making sure the shading frame is consistent with the view direction.
    float3x3 frame = evalGeom.frame;
    if (dot(frame[2], evalGeom.dir_in) < 0) {
        frame = -frame;
    }

    const float3 half_vector = normalize(evalGeom.dir_in + evalGeom.dir_out);
    const float n_dot_h = dot(frame[2], half_vector);
    const float n_dot_in = dot(frame[2], evalGeom.dir_in);
    const float n_dot_out = dot(frame[2], evalGeom.dir_out);
    if (n_dot_out <= 0 || n_dot_h <= 0) {
        return float3(0);
    }

    // dielectric layer:
    // F_o is the reflection percentage.
    const float F_o = FresnelDielectric(dot(half_vector, evalGeom.dir_out), material.eta);
    const float D = GTR2_NDF(n_dot_h, material.roughness);
    const float G = IsotropicGGX_Masking(to_local(frame, evalGeom.dir_in), material.roughness) *
                    IsotropicGGX_Masking(to_local(frame, evalGeom.dir_out), material.roughness);
    const float3 spec_contrib = material.Ks * (G * F_o * D) / (4 * n_dot_in * n_dot_out);
    // diffuse layer:
    // In order to reflect from the diffuse layer,
    // the photon needs to bounce through the dielectric layers twice.
    // The transmittance is computed by 1 - fresnel.
    const float F_i = FresnelDielectric(dot(half_vector, evalGeom.dir_in), material.eta);
    // Multiplying with Fresnels leads to an overly dark appearance at the
    // object boundaries. Disney BRDF proposes a fix to this -- we will implement this in problem set 1.
    const float3 diffuse_contrib = (1.f - F_o) * (1.f - F_i) / k_pi;
    return (spec_contrib + material.Kd * diffuse_contrib) * n_dot_out;
}

#endif // !_SRENDERER_ROUGHPLASTIC_HEADER_