#ifndef _SRENDERER_MMLT_DESCRIPTOR_SET_HEADER_
#define _SRENDERER_MMLT_DESCRIPTOR_SET_HEADER_

#include "mmlt_config.h"

struct SampleInfo {
    vec3    LCurrent;
    int     depth;
    vec2    pCurrent;
    vec2    padding;
};

layout(binding = 0, set = 2, r32f) coherent uniform image2DArray atomicRGB;
layout(binding = 1, set = 2, r32f) uniform image2D boostrapLuminance;
layout(binding = 2, set = 2) buffer _PSSSampleStreamBuffer  { vec4 sampleStreams[metroplis_buffer_width][metroplis_buffer_height][num_states_vec4]; };
layout(binding = 3, set = 2) buffer _PSSSampleInfoBuffer    { SampleInfo sampleInfos[metroplis_buffer_width][metroplis_buffer_height]; };
layout(binding = 4, set = 2) buffer _PSSNeuralStreamBuffer  { vec4 neuralStreams[metroplis_buffer_width][metroplis_buffer_height][num_states_vec4]; };
layout(binding = 5, set = 2) uniform sampler2D boostrapImportMIP;
layout(binding = 6, set = 2) uniform sampler2D testIMG;
#endif