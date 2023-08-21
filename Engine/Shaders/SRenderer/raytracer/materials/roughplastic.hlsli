#ifndef _SRENDERER_SPT_MATERIAL_ROUGHPLASTIC_HEADER_
#define _SRENDERER_SPT_MATERIAL_ROUGHPLASTIC_HEADER_

#include "../spt_interface.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/scene_descriptor_set.hlsli"

/**
 * Evaluate the RoughPlastic BSDF for the given query.
 * @param cBSDFEvalQuery The query to evaluate.
 */
[shader("callable")]
void EvalRoughPlastic(inout_ref(BSDFEvalQuery) cBSDFEvalQuery) {
    cBSDFEvalQuery.bsdf = float3(0);
    if (dot(cBSDFEvalQuery.geometric_normal, cBSDFEvalQuery.dir_in) < 0 ||
        dot(cBSDFEvalQuery.geometric_normal, cBSDFEvalQuery.dir_out) < 0) {
        // No light below the surface
        return;
    }
    // Making sure the shading frame is consistent with the view direction.
    float3x3 frame = cBSDFEvalQuery.frame;
    if (dot(frame[2], cBSDFEvalQuery.dir_in) < 0) {
        frame = -frame;
    }
    
    const float3 half_vector = normalize(cBSDFEvalQuery.dir_in + cBSDFEvalQuery.dir_out);
    const float n_dot_h = dot(frame[2], half_vector);
    const float n_dot_in = dot(frame[2], cBSDFEvalQuery.dir_in);
    const float n_dot_out = dot(frame[2], cBSDFEvalQuery.dir_out);
    if (n_dot_out <= 0 || n_dot_h <= 0) {
        return;
    }

    // Evaluate bsdf
    const MaterialInfo material = materials[cBSDFEvalQuery.mat_id];
    const float3 texAlbedo = textures[material.baseOrDiffuseTextureIndex]
                                 .Sample(cBSDFEvalQuery.uv, 0) .xyz;
    const float3 Kd = material.baseOrDiffuseColor * texAlbedo;
    const float3 Ks = material.specularColor;
    const float eta = material.transmissionFactor;
    float roughness = material.roughness;

    // Clamp roughness to avoid numerical issues.
    roughness = clamp(roughness, 0.01f, 1.f);
    // dielectric layer:
    // F_o is the reflection percentage.
    const float F_o = FresnelDielectric(dot(half_vector, cBSDFEvalQuery.dir_out), eta);
    const float D = GTR2_NDF(n_dot_h, roughness);
    const float G = IsotropicGGX_Masking(to_local(frame, cBSDFEvalQuery.dir_in), roughness) *
                    IsotropicGGX_Masking(to_local(frame, cBSDFEvalQuery.dir_out), roughness);
    const float3 spec_contrib = Ks * (G * F_o * D) / (4 * n_dot_in * n_dot_out);
    // diffuse layer:
    // In order to reflect from the diffuse layer,
    // the photon needs to bounce through the dielectric layers twice.
    // The transmittance is computed by 1 - fresnel.
    const float F_i = FresnelDielectric(dot(half_vector, cBSDFEvalQuery.dir_in), eta);
    // Multiplying with Fresnels leads to an overly dark appearance at the
    // object boundaries. Disney BRDF proposes a fix to this -- we will implement this in problem set 1.
    const float3 diffuse_contrib = Kd * (1.f - F_o) * (1.f - F_i) / k_pi;
    cBSDFEvalQuery.bsdf = (spec_contrib + diffuse_contrib) * n_dot_out;
    return;
}

/**
 * Sample the RoughPlastic BSDF for the given query.
 * @param cBSDFSampleQuery The query to evaluate.
 */
[shader("callable")]
void SampleRoughPlastic(inout_ref(BSDFSampleQuery) cBSDFSampleQuery) {
    cBSDFSampleQuery.pdf_out = 0;
    cBSDFSampleQuery.dir_out = float3(0);

    if (dot(cBSDFSampleQuery.geometric_normal, cBSDFSampleQuery.dir_in) < 0 ||
        dot(cBSDFSampleQuery.geometric_normal, cBSDFSampleQuery.dir_out) < 0) {
        // No light below the surface
        return;
    }
    // Making sure the shading frame is consistent with the view direction.
    float3x3 frame = cBSDFSampleQuery.frame;
    if (dot(frame[2], cBSDFSampleQuery.dir_in) < 0) {
        frame = -frame;
    }

    // Evaluate bsdf
    const MaterialInfo material = materials[cBSDFSampleQuery.mat_id];
    const float3 texAlbedo = textures[material.baseOrDiffuseTextureIndex]
                                 .Sample(cBSDFSampleQuery.uv, 0) .xyz;
    const float3 Kd = material.baseOrDiffuseColor * texAlbedo;
    const float3 Ks = material.specularColor;
    float lS = luminance(Ks);
    float lR = luminance(Kd);
    float roughness = material.roughness;
    // Clamp roughness to avoid numerical issues.
    roughness = clamp(roughness, 0.01f, 1.f);
    float alpha = roughness * roughness;

    if (lS + lR <= 0) {
        return;
    }
    float spec_prob = lS / (lS + lR);
    if (cBSDFSampleQuery.rnd_w < spec_prob) {
        // Sample from the specular lobe.
        // Convert the incoming direction to local coordinates
        float3 local_dir_in = to_local(frame, cBSDFSampleQuery.dir_in);
        float3 local_micro_normal =
            SampleVisibleNormals(local_dir_in, alpha, cBSDFSampleQuery.rnd_uv);
        // Transform the micro normal to world space
        float3 half_vector = to_world(frame, local_micro_normal);
        // Reflect over the world space normal
        float3 reflected = normalize(-cBSDFSampleQuery.dir_in 
            + 2 * dot(cBSDFSampleQuery.dir_in, half_vector) * half_vector);
        cBSDFSampleQuery.dir_out = reflected;
        
    }
    else {
        cBSDFSampleQuery.dir_out = to_world(
            frame,
            CosineWeightedHemisphereSample(
                float3(0, 0, 1),
                cBSDFSampleQuery.rnd_uv));
    }
    // We use the reflectance to determine whether to choose specular sampling lobe or diffuse.
    float diff_prob = 1 - spec_prob;
    const float3 half_vector = normalize(cBSDFSampleQuery.dir_in + cBSDFSampleQuery.dir_out);
    const float n_dot_h = dot(frame[2], half_vector);
    const float n_dot_in = dot(frame[2], cBSDFSampleQuery.dir_in);
    const float n_dot_out = dot(frame[2], cBSDFSampleQuery.dir_out);
    if (n_dot_out <= 0 || n_dot_h <= 0) {
        return;
    }
    // For the specular lobe, we use the ellipsoidal sampling from Heitz 2018
    // "Sampling the GGX Distribution of Visible Normals"
    // https://jcgt.org/published/0007/04/01/
    // this importance samples smith_masking(cos_theta_in) * GTR2(cos_theta_h, roughness) * cos_theta_out
    float G = IsotropicGGX_Masking(to_local(frame, cBSDFSampleQuery.dir_in), roughness);
    float D = GTR2_NDF(n_dot_h, roughness);
    // (4 * cos_theta_v) is the Jacobian of the reflectiokn
    spec_prob *= (G * D) / (4 * n_dot_in);
    // For the diffuse lobe, we importance sample cos_theta_out
    diff_prob *= n_dot_out / k_pi;
    cBSDFSampleQuery.pdf_out = spec_prob + diff_prob;
    return;
}

/**
 * Pdf of sampling the RoughPlastic BSDF for the given query.
 * @param cBSDFEvalQuery The query to evaluate.
 */
[shader("callable")]
void PdfRoughPlastic(inout_ref(BSDFSamplePDFQuery) cBSDFSamplePDFQuery) {
    cBSDFSamplePDFQuery.pdf = 0.f;
    if (dot(cBSDFSamplePDFQuery.geometric_normal, cBSDFSamplePDFQuery.dir_in) < 0 ||
        dot(cBSDFSamplePDFQuery.geometric_normal, cBSDFSamplePDFQuery.dir_out) < 0) {
        // No light below the surface
        return;
    }
    // Making sure the shading frame is consistent with the view direction.
    float3x3 frame = cBSDFSamplePDFQuery.frame;
    if (dot(frame[2], cBSDFSamplePDFQuery.dir_in) < 0) {
        frame = -frame;
    }

    float3 half_vector = normalize(cBSDFSamplePDFQuery.dir_in + cBSDFSamplePDFQuery.dir_out);
    const float n_dot_in = dot(frame[2], cBSDFSamplePDFQuery.dir_in);
    const float n_dot_out = dot(frame[2], cBSDFSamplePDFQuery.dir_out);
    const float n_dot_h = dot(frame[2], half_vector);
    if (n_dot_out <= 0 || n_dot_h <= 0) {
        return;
    }

    // Evaluate bsdf
    const MaterialInfo material = materials[cBSDFSamplePDFQuery.mat_id];
    const float3 texAlbedo = textures[material.baseOrDiffuseTextureIndex]
                                 .Sample(cBSDFSamplePDFQuery.uv, 0) .xyz;
    const float3 Kd = material.baseOrDiffuseColor * texAlbedo;
    const float3 Ks = material.specularColor;
    const float lS = luminance(Ks);
    const float lR = luminance(Kd);
    if (lS + lR <= 0) {
        return;
    }
    float roughness = material.roughness;
    // Clamp roughness to avoid numerical issues.
    roughness = clamp(roughness, 0.01f, 1.f);
    // We use the reflectance to determine whether to choose specular sampling lobe or diffuse.
    float spec_prob = lS / (lS + lR);
    float diff_prob = 1 - spec_prob;
    // For the specular lobe, we use the ellipsoidal sampling from Heitz 2018
    // "Sampling the GGX Distribution of Visible Normals"
    // https://jcgt.org/published/0007/04/01/
    // this importance samples smith_masking(cos_theta_in) * GTR2(cos_theta_h, roughness) * cos_theta_out
    const float G = IsotropicGGX_Masking(to_local(frame, cBSDFSamplePDFQuery.dir_in), roughness);
    const float D = GTR2_NDF(n_dot_h, roughness);
    // (4 * cos_theta_v) is the Jacobian of the reflectiokn
    spec_prob *= (G * D) / (4 * n_dot_in);
    // For the diffuse lobe, we importance sample cos_theta_out
    diff_prob *= n_dot_out / k_pi;
    cBSDFSamplePDFQuery.pdf = spec_prob + diff_prob;
    return;
}

#endif // !_SRENDERER_SPT_MATERIAL_ROUGHPLASTIC_HEADER_