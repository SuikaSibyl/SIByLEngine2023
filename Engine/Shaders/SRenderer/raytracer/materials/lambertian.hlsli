#ifndef _SRENDERER_SPT_MATERIAL_LAMBERTIAN_HEADER_
#define _SRENDERER_SPT_MATERIAL_LAMBERTIAN_HEADER_

#include "../../include/diff_descriptor_set.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/scene_descriptor_set.hlsli"
#include "../spt_interface.hlsli"

/**
 * Evaluate the Lambertian BSDF for the given query.
 * @param cBSDFEvalQuery The query to evaluate.
 */
[shader("callable")]
void EvalLambertian(inout_ref(BSDFEvalQuery) cBSDFEvalQuery) {
    if (dot(cBSDFEvalQuery.geometric_normal, cBSDFEvalQuery.dir_in) < 0 ||
        dot(cBSDFEvalQuery.geometric_normal, cBSDFEvalQuery.dir_out) < 0) {
        // No light below the surface
        cBSDFEvalQuery.bsdf = float3(0);
        cBSDFEvalQuery.dir_out = float3(0); // override specular
        return;
    }
    // Making sure the shading frame is consistent with the view direction.
    float3x3 frame = cBSDFEvalQuery.frame;
    const QueryBitfield bitfield = UnpackQueryBitfield(cBSDFEvalQuery.misc_flag);

    // Evaluate bsdf
    float3 albedo;
    if (cBSDFEvalQuery.mat_id == 0xFFFFFFFF) {
        albedo = UnpackRGBE(asuint(cBSDFEvalQuery.bsdf.x));
    }
    else {
        const MaterialInfo material = materials[cBSDFEvalQuery.mat_id];
        const float3 texAlbedo = textures[material.baseOrDiffuseTextureIndex]
                                     .Sample(cBSDFEvalQuery.uv, 0) .xyz;
        albedo = material.baseOrDiffuseColor * texAlbedo;
    }
    const float3 demodulate = saturate(dot(frame[2], cBSDFEvalQuery.dir_out)) / k_pi;
    cBSDFEvalQuery.bsdf = bitfield.split_query ? demodulate : albedo * demodulate;
    cBSDFEvalQuery.dir_out = float3(0); // override specular
}

/**
 * Sample the Lambertian BSDF for the given query.
 * @param cBSDFSampleQuery The query to evaluate.
 */
[shader("callable")]
void SampleLambertian(inout_ref(BSDFSampleQuery) cBSDFSampleQuery) {
    // Check the direction
    if (dot(cBSDFSampleQuery.geometric_normal, cBSDFSampleQuery.dir_in) < 0) {
        // No light below the surface
        cBSDFSampleQuery.dir_out = float3(0);
        cBSDFSampleQuery.pdf_out = 0.f;
        return;
    }
    // Making sure the shading frame is consistent with the view direction.
    float3x3 frame = cBSDFSampleQuery.frame;
    // Sample bsdf
    cBSDFSampleQuery.dir_out = to_world(
        frame,
        CosineWeightedHemisphereSample(
            float3(0, 0, 1),
            cBSDFSampleQuery.rnd_uv));
    cBSDFSampleQuery.pdf_out = saturate(dot(frame[2], cBSDFSampleQuery.dir_out)) / k_pi;
}

/**
 * Pdf of sampling the Lambertian BSDF for the given query.
 * @param cBSDFEvalQuery The query to evaluate.
 */
[shader("callable")]
void PdfLambertian(inout_ref(BSDFSamplePDFQuery) cBSDFSamplePDFQuery) {
    if (dot(cBSDFSamplePDFQuery.geometric_normal, cBSDFSamplePDFQuery.dir_in) < 0 ||
        dot(cBSDFSamplePDFQuery.geometric_normal, cBSDFSamplePDFQuery.dir_out) < 0) {
        // No light below the surface
        cBSDFSamplePDFQuery.pdf = 0.f;
        return;
    }
    // Making sure the shading frame is consistent with the view direction.
    float3x3 frame = cBSDFSamplePDFQuery.frame;
    cBSDFSamplePDFQuery.pdf = saturate(dot(frame[2], cBSDFSamplePDFQuery.dir_out)) / k_pi;
}

/**
 * Evaluate the Lambertian BSDF for the given query with consider of differentiable.
 * At the same time, evaluate the gradient for every differentiable parameter.
 * @param cBSDFEvalQuery The query to evaluate.
 */
[shader("callable")]
void EvalDiffLambertian(inout_ref(BSDFEvalDiffQuery) cBSDFEvalDiffQuery) {
    if (dot(cBSDFEvalDiffQuery.geometric_normal, cBSDFEvalDiffQuery.dir_in) < 0 ||
        dot(cBSDFEvalDiffQuery.geometric_normal, cBSDFEvalDiffQuery.dir_out) < 0) {
        // No light below the surface
        cBSDFEvalDiffQuery.bsdf = float3(0);
        cBSDFEvalDiffQuery.dir_out = float3(0); // override specular
        return;
    }
    // Making sure the shading frame is consistent with the view direction.
    float3x3 frame = cBSDFEvalDiffQuery.frame;
    const QueryBitfield bitfield = UnpackQueryBitfield(cBSDFEvalDiffQuery.misc_flag);

    // Evaluate bsdf
    float3 albedo;
    int albedo_tex_id = -1;
    if (cBSDFEvalDiffQuery.mat_id == 0xFFFFFFFF) {
        albedo = UnpackRGBE(asuint(cBSDFEvalDiffQuery.bsdf.x));
    }
    else {
        const MaterialInfo material = materials[cBSDFEvalDiffQuery.mat_id];
        albedo_tex_id = material.baseOrDiffuseTextureIndex;
        const float3 texAlbedo = textures[albedo_tex_id].Sample(cBSDFEvalDiffQuery.uv, 0).xyz;
        albedo = material.baseOrDiffuseColor * texAlbedo;
    }
    const float3 demodulate = saturate(dot(frame[2], cBSDFEvalDiffQuery.dir_out)) / k_pi;
    cBSDFEvalDiffQuery.bsdf = bitfield.split_query ? demodulate : albedo * demodulate;
    cBSDFEvalDiffQuery.dir_out = float3(0); // override specular

    // if the albedo texture if a differentiable resource
    // then we need to evaluate the gradient for albedo
    if (albedo_tex_id != -1) {
        int diff_id_albedo = DiffableTextureIndices[albedo_tex_id];
        const DiffResourceDesc diff_desc = DiffResourcesDescs[diff_id_albedo];
        float3 albedo_gradient = cBSDFEvalDiffQuery.adjoint_gradient * demodulate;
        const int width = (diff_desc.data_extend >> 0) & 0xFFFF;
        const int height = (diff_desc.data_extend >> 16) & 0xFFFF;
        if (any(isnan(albedo_gradient) || isinf(albedo_gradient))) {
        } else {
            float lod = 0.5 * log2(width * height) + cBSDFEvalDiffQuery.lod;
            lod = clamp(lod, 0.f, 15.f);
            // trilinear_gradient_spatting(albedo_gradient, float3(cBSDFEvalDiffQuery.uv, lod), int2(width, height), 0);
            InterlockedAddGradTex(albedo_gradient, albedo_tex_id, float3(cBSDFEvalDiffQuery.uv, lod));
        }
    }
}

#endif // !_SRENDERER_SPT_MATERIAL_LAMBERTIAN_HEADER_