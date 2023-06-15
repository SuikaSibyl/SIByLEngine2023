#ifndef _RESTIR_DI_PIPELINE_LAYTOU_H_
#define _RESTIR_DI_PIPELINE_LAYTOU_H_

#include "../../common/restir_di_def.h"

struct UniformStruct {
    PlanarViewData view;
    PlanarViewData prevView;
    ReSTIR_DI_ResamplingRuntimeParameters runtimeParams;
    uint initialOutputBufferIndex;
    uint frameIndex;

    uint numPrimaryRegirSamples;
    uint numPrimaryLocalLightSamples;
    uint numPrimaryInfiniteLightSamples;
    uint numPrimaryEnvironmentSamples;
    uint numPrimaryBrdfSamples;
    float brdfCutoff;
    
    uint enableInitialVisibility;
};

layout(binding =  0, set = 2, scalar) uniform _Uniforms { UniformStruct gUniform; };

// previous gbuffer
layout(binding =  1, set = 2)   uniform  sampler2D t_PrevGBufferDepth;
layout(binding =  2, set = 2)   uniform usampler2D t_PrevGBufferNormals;
layout(binding =  3, set = 2)   uniform usampler2D t_PrevGBufferGeoNormals;
layout(binding =  4, set = 2)   uniform  sampler2D t_PrevGBufferDiffuseAlbedo;
layout(binding =  5, set = 2)   uniform  sampler2D t_PrevGBufferSpecularRough;
// current gbuffer
layout(binding =  6, set = 2)   uniform  sampler2D t_GBufferDepth;
layout(binding =  7, set = 2)   uniform usampler2D t_GBufferNormals;
layout(binding =  8, set = 2)   uniform usampler2D t_GBufferGeoNormals;
layout(binding =  9, set = 2)   uniform  sampler2D t_GBufferDiffuseAlbedo;
layout(binding = 10, set = 2)   uniform  sampler2D t_GBufferSpecularRough;

layout(binding = 11, set = 3, scalar) buffer _ReservoirBuffer { DIReservoirPacked t_Reservoir[]; };

#define RTXDI_LIGHT_RESERVOIR_BUFFER t_Reservoir
#include "../../common/restir_di.glsl"

#endif // !_RESTIR_DI_PIPELINE_LAYTOU_H_