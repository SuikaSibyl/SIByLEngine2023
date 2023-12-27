#ifndef _SRENDERER_COMMON_DIFF_SCENE_IO_HEADER_
#define _SRENDERER_COMMON_DIFF_SCENE_IO_HEADER_

#include "diff_descriptor_set.hlsli"
#include "scene_descriptor_set.hlsli"

[BackwardDerivative(bwd_LoadGeometryTransformAD)]
float4[3] LoadGeometryTransformAD(in GeometryInfo geometry, int gradOffset, int hash = 0) {
    return geometry.transform;
}

void bwd_LoadGeometryTransformAD(in GeometryInfo geometry, int gradOffset, int hash = 0, in float4[3] grad) {
    InterlockedAddGrad(DiffInfoType::DIFF_INFO_TYPE_TRANSFORM, gradOffset, grad[0], hash);
    InterlockedAddGrad(DiffInfoType::DIFF_INFO_TYPE_TRANSFORM, gradOffset + 1, grad[1], hash);
    InterlockedAddGrad(DiffInfoType::DIFF_INFO_TYPE_TRANSFORM, gradOffset + 2, grad[2], hash);
}



#endif // !_SRENDERER_COMMON_DIFF_SCENE_IO_HEADER_