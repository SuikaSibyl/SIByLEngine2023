#ifndef _SRENDERER_SPT_MATERIAL_ROUGHDIELECTRIC_HEADER_
#define _SRENDERER_SPT_MATERIAL_ROUGHDIELECTRIC_HEADER_

#include "../../include/common/sampling.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../spt_interface.hlsli"

/**
 * Evaluate the RoughDielectric BSDF for the given query.
 * @param cBSDFEvalQuery The query to evaluate.
 */
[shader("callable")]
void EvalRoughDielectric(inout_ref(BSDFEvalQuery) cBSDFEvalQuery) {
    float bsdf_eta;
    float3 Kt;
    float3 Ks;
    float roughness;
    // First load the material info
    // -------------------------------------------------------------
    if (cBSDFEvalQuery.mat_id == 0xFFFFFFFF) {
        // info is already packed in the query
        Kt = UnpackRGBE(asuint(cBSDFEvalQuery.bsdf.x));
        Ks = UnpackRGBE(asuint(cBSDFEvalQuery.uv.y));
        bsdf_eta = cBSDFEvalQuery.bsdf.y;
        roughness = cBSDFEvalQuery.uv.x;
    } else { // load info from the material buffer
        const MaterialInfo material = materials[cBSDFEvalQuery.mat_id];
        const float3 texAlbedo = textures[material.baseOrDiffuseTextureIndex]
                                     .Sample(cBSDFEvalQuery.uv, 0) .xyz;
        Kt = material.baseOrDiffuseColor * texAlbedo;
        Ks = material.specularColor;
        bsdf_eta = material.transmissionFactor;
        roughness = material.roughness;
    }

    // Load dir in/out for simplicity
    const QueryBitfield bitfield = UnpackQueryBitfield(cBSDFEvalQuery.misc_flag);
    const float3 dir_in = cBSDFEvalQuery.dir_in;
    const float3 dir_out = cBSDFEvalQuery.dir_out;
    const float3 geometry_normal = bitfield.face_forward
        ? +cBSDFEvalQuery.geometric_normal
        : -cBSDFEvalQuery.geometric_normal;
    const bool reflected = dot(geometry_normal, dir_in) *
                           dot(geometry_normal, dir_out) > 0;

    // Flip the shading frame if it is inconsistent with the geometry normal
    float3x3 frame = cBSDFEvalQuery.frame;
    if (dot(frame[2], dir_in) * dot(geometry_normal, dir_in) < 0) {
        frame = -frame;
    }
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    const float eta = dot(geometry_normal, dir_in) > 0 ? bsdf_eta : 1. / bsdf_eta;
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
    
    // Clamp roughness to avoid numerical issues.
    roughness = clamp(roughness, 0.01, 1);

    // Compute F / D / G
    // Note that we use the incoming direction
    // for evaluating the Fresnel reflection amount.
    // We can also use outgoing direction -- then we would need to
    // use 1/bsdf.eta and we will get the same result.
    // However, using the incoming direction allows
    // us to use F to decide whether to reflect or refract during sampling.
    float h_dot_in = dot(half_vector, dir_in);
    float F = FresnelDielectric(h_dot_in, eta);
    float D = GTR2_NDF(dot(frame[2], half_vector), roughness);
    float G = IsotropicGGX_Masking(to_local(frame, dir_in), roughness) *
              IsotropicGGX_Masking(to_local(frame, dir_out), roughness);

    if (reflected) {
        cBSDFEvalQuery.bsdf = Ks * (F * D * G) / (4 * abs(dot(frame[2], dir_in)));
    } else {
        // Snell-Descartes law predicts that the light will contract/expand
        // due to the different index of refraction. So the normal BSDF needs
        // to scale with 1/eta^2. However, the "adjoint" of the BSDF does not have
        // the eta term. This is due to the non-reciprocal nature of the index of refraction:
        // f(wi -> wo) / eta_o^2 = f(wo -> wi) / eta_i^2
        // thus f(wi -> wo) = f(wo -> wi) (eta_o / eta_i)^2
        // The adjoint of a BSDF is defined as swapping the parameter, and
        // this cancels out the eta term.
        // See Chapter 5 of Eric Veach's thesis "Robust Monte Carlo Methods for Light Transport Simulation"
        // for more details.
        float eta_factor = (bitfield.transport_mode == enum_transport_radiance) ? (1 / (eta * eta)) : 1;
        float h_dot_out = dot(half_vector, dir_out);
        float sqrt_denom = h_dot_in + eta * h_dot_out;
        // Very complicated BSDF. See Walter et al.'s paper for more details.
        // "Microfacet Models for Refraction through Rough Surfaces"
        cBSDFEvalQuery.bsdf = Kt * (eta_factor * (1 - F) * D * G * eta * eta * abs(h_dot_out * h_dot_in)) /
                              (abs(dot(frame[2], dir_in)) * sqrt_denom * sqrt_denom);
    }
    return;
}

/**
 * Sample the RoughDielectric BSDF for the given query.
 * @param cBSDFSampleQuery The query to evaluate.
 */
[shader("callable")]
void SampleRoughDielectric(inout_ref(BSDFSampleQuery) cBSDFSampleQuery) {
    float bsdf_eta;
    float3 Kt;
    float3 Ks;
    float roughness;
    // First load the material info
    // -------------------------------------------------------------
    if (cBSDFSampleQuery.mat_id == 0xFFFFFFFF) {
        // info is already packed in the query
        // info is already packed in the query
        Kt = UnpackRGBE(asuint(cBSDFSampleQuery.dir_out.x));
        Ks = UnpackRGBE(asuint(cBSDFSampleQuery.uv.y));
        bsdf_eta = cBSDFSampleQuery.dir_out.y;
        roughness = cBSDFSampleQuery.uv.x;
    } else { // load info from the material buffer
        const MaterialInfo material = materials[cBSDFSampleQuery.mat_id];
        const float3 texAlbedo = textures[material.baseOrDiffuseTextureIndex]
                                     .Sample(cBSDFSampleQuery.uv, 0) .xyz;
        Kt = material.baseOrDiffuseColor * texAlbedo;
        Ks = material.specularColor;
        bsdf_eta = material.transmissionFactor;
        roughness = material.roughness;
    }

    // Load dir in/out for simplicity
    const QueryBitfield bitfield = UnpackQueryBitfield(cBSDFSampleQuery.misc_flag);
    const float3 dir_in = cBSDFSampleQuery.dir_in;
    const float3 geometry_normal = bitfield.face_forward
                                       ? +cBSDFSampleQuery.geometric_normal
                                       : -cBSDFSampleQuery.geometric_normal;

    // Flip the shading frame if it is inconsistent with the geometry normal
    float3x3 frame = cBSDFSampleQuery.frame;
    if (dot(frame[2], dir_in) * dot(geometry_normal, dir_in) < 0) {
        frame = -frame;
    }
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    const float eta = dot(geometry_normal, dir_in) > 0 ? bsdf_eta : 1. / bsdf_eta;
    // Clamp roughness to avoid numerical issues.
    roughness = clamp(roughness, 0.01, 1);
    // Sample a micro normal and transform it to world space -- this is our half-vector.
    const float alpha = roughness * roughness;
    const float3 local_dir_in = to_local(frame, dir_in);
    const float3 local_micro_normal =
        SampleVisibleNormals(local_dir_in, alpha, cBSDFSampleQuery.rnd_uv);

    float3 half_vector = to_world(frame, local_micro_normal);
    // Flip half-vector if it's below surface
    if (dot(half_vector, frame[2]) < 0) {
        half_vector = -half_vector;
    }

    // Now we need to decide whether to reflect or refract.
    // We do this using the Fresnel term.
    const float h_dot_in = dot(half_vector, dir_in);
    const float F = FresnelDielectric(h_dot_in, eta);

    if (cBSDFSampleQuery.rnd_w <= F) {
        // Reflection
        float3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
        // set eta to 0 since we are not transmitting
        cBSDFSampleQuery.dir_out = reflected;
    } else {
        // Refraction
        // https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
        // (note that our eta is eta2 / eta1, and l = -dir_in)
        float h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
        if (h_dot_out_sq <= 0) {
            // Total internal reflection
            // This shouldn't really happen, as F will be 1 in this case.
            cBSDFSampleQuery.pdf_out = 0.f;
            return;
        }
        // flip half_vector if needed
        if (h_dot_in < 0) {
            half_vector = -half_vector;
        }
        float h_dot_out = sqrt(h_dot_out_sq);
        float3 refracted = -dir_in / eta + (abs(h_dot_in) / eta - h_dot_out) * half_vector;
        cBSDFSampleQuery.dir_out = refracted;
    }
    
    const float3 dir_out = cBSDFSampleQuery.dir_out;
    const bool reflected = dot(geometry_normal, dir_in) *
                           dot(geometry_normal, dir_out) > 0;
    // We sample the visible normals, also we use F to determine
    // whether to sample reflection or refraction
    // so PDF ~ F * D * G_in for reflection, PDF ~ (1 - F) * D * G_in for refraction.
    float D = GTR2_NDF(dot(half_vector, frame[2]), roughness);
    float G_in = IsotropicGGX_Masking(to_local(frame, dir_in), roughness);
    if (reflected) {
        cBSDFSampleQuery.pdf_out = (F * D * G_in) / (4 * abs(dot(frame[2], dir_in)));
    } else {
        float h_dot_out = dot(half_vector, dir_out);
        float sqrt_denom = h_dot_in + eta * h_dot_out;
        float dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
        cBSDFSampleQuery.pdf_out = (1 - F) * D * G_in * abs(dh_dout * h_dot_in / dot(frame[2], dir_in));
    }
}

/**
 * Pdf of sampling the RoughDielectric BSDF for the given query.
 * @param cBSDFEvalQuery The query to evaluate.
 */
[shader("callable")]
void PdfRoughDielectric(inout_ref(BSDFSamplePDFQuery) cBSDFSamplePDFQuery) {
    // First load the material info
    // -------------------------------------------------------------
    float3 Kd;
    float3 Ks;
    float roughness;
    float bsdf_eta;
    if (cBSDFSamplePDFQuery.mat_id == 0xFFFFFFFF) {
        // info is already packed in the query
        Kd = UnpackRGBE(asuint(cBSDFSamplePDFQuery.packedInfo0));
        Ks = UnpackRGBE(asuint(cBSDFSamplePDFQuery.uv.y));
        roughness = cBSDFSamplePDFQuery.uv.x;
        bsdf_eta = cBSDFSamplePDFQuery.packedInfo1;
    }
    else {
        // load info from the material buffer
        const MaterialInfo material = materials[cBSDFSamplePDFQuery.mat_id];
        const float3 texAlbedo = textures[material.baseOrDiffuseTextureIndex]
                                     .Sample(cBSDFSamplePDFQuery.uv, 0) .xyz;
        Kd = material.baseOrDiffuseColor * texAlbedo;
        Ks = material.specularColor;
        roughness = material.roughness;
        bsdf_eta = material.transmissionFactor;
    }

    // Load dir in/out for simplicity
    const QueryBitfield bitfield = UnpackQueryBitfield(cBSDFSamplePDFQuery.misc_flag);
    const float3 dir_in = cBSDFSamplePDFQuery.dir_in;
    const float3 dir_out = cBSDFSamplePDFQuery.dir_out;
    const float3 geometry_normal = bitfield.face_forward
                                       ? +cBSDFSamplePDFQuery.geometric_normal
                                       : -cBSDFSamplePDFQuery.geometric_normal;

    const bool reflected = dot(geometry_normal, dir_in) *
                           dot(geometry_normal, dir_out) > 0;

    // Flip the shading frame if it is inconsistent with the geometry normal
    float3x3 frame = cBSDFSamplePDFQuery.frame;
    if (dot(frame[2], dir_in) * dot(geometry_normal, dir_in) < 0) {
        frame = -frame;
    }
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    const float eta = dot(geometry_normal, dir_in) > 0 ? bsdf_eta : 1. / bsdf_eta;
    
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

    // Clamp roughness to avoid numerical issues.
    roughness = clamp(roughness, 0.01, 1);

    // We sample the visible normals, also we use F to determine
    // whether to sample reflection or refraction
    // so PDF ~ F * D * G_in for reflection, PDF ~ (1 - F) * D * G_in for refraction.
    float h_dot_in = dot(half_vector, dir_in);
    float F = FresnelDielectric(h_dot_in, eta);
    float D = GTR2_NDF(dot(half_vector, frame[2]), roughness);
    float G_in = IsotropicGGX_Masking(to_local(frame, dir_in), roughness);
    if (reflected) {
        cBSDFSamplePDFQuery.pdf = (F * D * G_in) / (4 * abs(dot(frame[2], dir_in)));
    } else {
        float h_dot_out = dot(half_vector, dir_out);
        float sqrt_denom = h_dot_in + eta * h_dot_out;
        float dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
        cBSDFSamplePDFQuery.pdf = (1 - F) * D * G_in * abs(dh_dout * h_dot_in / dot(frame[2], dir_in));
    }
}

#endif // !_SRENDERER_SPT_MATERIAL_ROUGHDIELECTRIC_HEADER_