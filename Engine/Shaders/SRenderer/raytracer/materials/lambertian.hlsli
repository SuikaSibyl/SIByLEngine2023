#ifndef _SRENDERER_SPT_MATERIAL_LAMBERTIAN_HEADER_
#define _SRENDERER_SPT_MATERIAL_LAMBERTIAN_HEADER_

#include "../spt_interface.hlsli"
#include "../../include/common/sampling.hlsli"
#include "../../include/scene_descriptor_set.hlsli"

/**
 * Evaluate the Lambertian BSDF for the given query.
 * @param cBSDFEvalQuery The query to evaluate.
 */
[shader("callable")]
void EvalLambertian(inout_ref(BSDFEvalQuery) cBSDFEvalQuery) {
    // Making sure the shading frame is consistent with the view direction.
    float3x3 frame = cBSDFEvalQuery.frame;
    // if (dot(frame[2], cBSDFEvalQuery.dir_in) < 0) {
    //     frame = -frame;
    // }
    // Evaluate bsdf
    const MaterialInfo material = materials[cBSDFEvalQuery.mat_id];
    const float3 texAlbedo = textures[material.baseOrDiffuseTextureIndex]
                                 .Sample(cBSDFEvalQuery.uv, 0) .xyz;
    const float3 albedo = material.baseOrDiffuseColor * texAlbedo / k_pi;
    cBSDFEvalQuery.bsdf = saturate(dot(frame[2], cBSDFEvalQuery.dir_out)) * albedo;
}

/**
 * Sample the Lambertian BSDF for the given query.
 * @param cBSDFSampleQuery The query to evaluate.
 */
[shader("callable")]
void SampleLambertian(inout_ref(BSDFSampleQuery) cBSDFSampleQuery) {
    // Making sure the shading frame is consistent with the view direction.
    float3x3 frame = cBSDFSampleQuery.frame;
    if (dot(frame[2], cBSDFSampleQuery.dir_in) < 0) {
        frame = -frame;
    }
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
    // Making sure the shading frame is consistent with the view direction.
    float3x3 frame = cBSDFSamplePDFQuery.frame;
    if (dot(frame[2], cBSDFSamplePDFQuery.dir_in) < 0) {
        frame = -frame;
    }
    cBSDFSamplePDFQuery.pdf = saturate(dot(frame[2], cBSDFSamplePDFQuery.dir_out)) / k_pi;
}

#endif // !_SRENDERER_SPT_MATERIAL_LAMBERTIAN_HEADER_