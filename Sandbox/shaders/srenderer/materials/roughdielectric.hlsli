#ifndef _SRENDERER_ROUGHDIELECTRIC_HEADER_
#define _SRENDERER_ROUGHDIELECTRIC_HEADER_

#include "../../common/math.hlsli"
#include "../../common/microfacet.hlsli"
#include "../spt.hlsli"

struct RoughDielectricMaterial : IDifferentiable {
    float3 Kt;
    float eta;
    float3 Ks;
    float roughness;
};

[Differentiable]
float3 EvalRoughDielectric(
    RoughDielectricMaterial material,
    no_diff BSDFEvalGeometry evalGeom
) {
    const float3 dir_in = evalGeom.dir_in;
    const float3 dir_out = evalGeom.dir_out;
    const float3 geometry_normal = bitfield.face_forward
                                       ? +evalGeom.geometric_normal
                                       : -evalGeom.geometric_normal;
    const bool reflected = dot(geometry_normal, dir_in) * dot(geometry_normal, dir_out) > 0;

    // Flip the shading frame if it is inconsistent with the geometry normal
    float3x3 frame = evalGeom.frame;
    if (dot(frame[2], dir_in) * dot(geometry_normal, dir_in) < 0) {
        frame = -frame;
    }
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    const float eta = dot(geometry_normal, dir_in) > 0 ? material.eta : 1. / material.eta;
    // half vector
    float3 half_vector;
    if (reflected) {
        half_vector = normalize(dir_in + dir_out);
    } else {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_vector = normalize(dir_in + dir_out * eta);
    }

    // Flip half-vector if it's below surface
    if (dot(half_vector, frame[2]) < 0) {
        half_vector = -half_vector;
    }
    
    float h_dot_in = dot(half_vector, dir_in);
    float F = FresnelDielectric(h_dot_in, eta);
    float D = GTR2_NDF(dot(frame[2], half_vector), material.roughness);
    float G = IsotropicGGX_Masking(no_diff to_local(frame, dir_in), material.roughness) *
              IsotropicGGX_Masking(no_diff to_local(frame, dir_out), material.roughness);

    return material.Ks * (F * D * G) / (4 * abs(dot(frame[2], dir_in)));
}

#endif // !_SRENDERER_ROUGHDIELECTRIC_HEADER_