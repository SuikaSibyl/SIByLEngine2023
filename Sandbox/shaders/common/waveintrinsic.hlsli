#ifndef _SRENDERER_WAVEINTRINSIC_HLSLI_
#define _SRENDERER_WAVEINTRINSIC_HLSLI_

// matrix multiplication for 64x64 matrix and a 64x128 tensor
// assume the weights are already loaded into the shared memory
// and the matrix weights are already on the subgroup matrix objects,
// which is loaded by "__inline_wmma_load_64_64_matrices"
float glsl_subgroupClusteredAdd_IDX2(float value) {
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_basic : enable)");
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_clustered : enable)");
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_shuffle : enable)");
    __intrinsic_asm "subgroupClusteredAdd($0, 2)";
}

float2 glsl_subgroupClusteredAdd_IDX2_float2(float2 value) {
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_basic : enable)");
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_clustered : enable)");
    __requirePrelude(R"(#extension GL_KHR_shader_subgroup_shuffle : enable)");
    __intrinsic_asm "subgroupClusteredAdd($0, 2)";
}

#endif // _SRENDERER_WAVEINTRINSIC_HLSLI_