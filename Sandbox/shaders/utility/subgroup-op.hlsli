#ifndef _SRENDERER_SUBGROUP_HLSLI_HEADER_
#define _SRENDERER_SUBGROUP_HLSLI_HEADER_

float __inline_subgroupClusteredAdd2(float value) {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "subgroup-op.glsli")");
    __intrinsic_asm "glsl_subgroupClusteredAdd2($0)";
}

float __inline_subgroupShuffle(float value, uint index) {
    __requirePrelude(R"(#extension GL_GOOGLE_include_directive : enable)");
    __requirePrelude(R"(#include "subgroup-op.glsli")");
    __intrinsic_asm "glsl_subgroupShuffle($0, $1)";
}

#endif // _SRENDERER_SUBGROUP_HLSLI_HEADER_