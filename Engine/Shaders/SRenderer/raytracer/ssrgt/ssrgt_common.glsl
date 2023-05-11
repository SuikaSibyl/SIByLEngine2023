#ifndef _SRENDERER_COMMON_SSRGT_HEADER_
#define _SRENDERER_COMMON_SSRGT_HEADER_

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_nonuniform_qualifier : enable

#include "../../include/definitions/camera.h"

struct GlobalUniforms {
  CameraData cameraData;
};
layout(binding = 0, set = 0) uniform _GlobalUniforms  { GlobalUniforms globalUniform; };

#include "../include/utils/camera_model.h"
#include "../include/utils/ray_intersection.h"

layout(binding = 1, set = 0, scalar) buffer _VertexBuffer     { vec3 vertices[]; };
layout(binding = 2, set = 0, scalar) buffer _IndicesBuffer    { uint indices[]; };
layout(binding = 3, set = 0) uniform accelerationStructureEXT tlas;
layout(binding = 4, set = 0, rgba32f) uniform image2D storageImage;


struct UniformStruct {
    vec2    view_size;
    int     hiz_mip_levels;
    uint    max_iteration;
    int     strategy;
    int     sample_batch;
    uint    debug_ray_mode;
    float   max_thickness;
    uint    debug_mode;
    int	    mip_level;
    int     offset_steps;
    float   z_clamper;
    vec4    iDebugPos;
    float   z_min;
    float   z_range;
    int     is_depth;
    int     lightcut_mode;
    mat4    InvProjMat;
    mat4    ProjMat;
    mat4    TransInvViewMat;
};

layout(binding = 0, set = 1) uniform sampler2D base_color;
layout(binding = 1, set = 1) uniform sampler2D hi_z;
layout(binding = 2, set = 1) uniform sampler2D ws_normal;
layout(binding = 3, set = 1) uniform sampler2D importance_mip;
layout(binding = 4, set = 1) uniform sampler2D boundingbox_mip;
layout(binding = 5, set = 1) uniform sampler2D bbncpack_mip;
layout(binding = 6, set = 1) uniform sampler2D normalcone_mip;
layout(binding = 7, set = 1) uniform sampler2D di;
layout(binding = 8, set = 1, scalar) uniform _Uniforms { UniformStruct gUniform; };

struct RayPayload {
    vec3 position;
    bool hit;
    int triangleIndex;
};

#endif