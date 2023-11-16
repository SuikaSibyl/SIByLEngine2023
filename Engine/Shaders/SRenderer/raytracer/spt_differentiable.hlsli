#ifndef _SRENDERER_SPT_DIFFERENTIABLE_HEADER_
#define _SRENDERER_SPT_DIFFERENTIABLE_HEADER_

#include "../include/scene_descriptor_set.hlsli"
#include "../include/common/camera.hlsli"
#include "../include/common/cpp_compatible.hlsli"
#include "../include/common/math.hlsli"
#include "../include/common/random.hlsli"
#include "../include/common/raycast.hlsli"
#include "../include/common/shading.hlsli"
#include "../include/common/packing.hlsli"
#include "../include/raytracer_descriptor_set.hlsli"
#include "spt_interface.hlsli"

#define BSDF_EVAL_DIFF_OFFSET 9
#define BSDF_EVAL_DIFF_IDX(BSDF_ID) (BSDF_EVAL_DIFF_OFFSET + (BSDF_ID))

/**
 * Evaluate the BSDF.
 * @param hit The geometry hit.
 * @param dir_in The direction of the incoming ray.
 * @param dir_out The direction of the outgoing ray.
 * @param transport_mode The transport mode.
 * @return The evaluated BSDF value.
 */
float3 EvalBsdfDiff(
    in_ref(GeometryHit) hit,
    in_ref(float3) dir_in,
    in_ref(float3) dir_out,
    in_ref(float3) adjoint_gradient,
    out_ref(float3) debug,
    in const float lod_indicator = 0,
    in const uint transport_mode = 0
) {
    uint materialID = geometries[hit.geometryID].materialID;
    BSDFEvalDiffQuery cBSDFEvalDiffQuery;
    uint bsdf_type = materials[materialID].bsdfID;
    cBSDFEvalDiffQuery.dir_in = dir_in;
    cBSDFEvalDiffQuery.dir_out = dir_out;
    cBSDFEvalDiffQuery.mat_id = materialID;
    cBSDFEvalDiffQuery.geometric_normal = hit.geometryNormal;
    cBSDFEvalDiffQuery.uv = hit.texcoord;
    cBSDFEvalDiffQuery.frame = createONB(hit.shadingNormal);
    cBSDFEvalDiffQuery.adjoint_gradient = adjoint_gradient;
    cBSDFEvalDiffQuery.lod = lod_indicator;
    QueryBitfield flag;
    flag.transport_mode = transport_mode;
    flag.face_forward = IsFaceForward(hit);
    flag.split_query = false;
    cBSDFEvalDiffQuery.misc_flag = PackQueryBitfield(flag);
    CallShader(BSDF_EVAL_DIFF_IDX(bsdf_type), cBSDFEvalDiffQuery);
    debug = cBSDFEvalDiffQuery.dir_out;
    return cBSDFEvalDiffQuery.bsdf;
}

#endif // !_SRENDERER_SPT_DIFFERENTIABLE_HEADER_