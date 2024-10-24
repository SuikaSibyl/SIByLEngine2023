#version 460
#extension GL_GOOGLE_include_directive : enable

#include "../include/common_vert.h"
#include "../../include/common/glsl_compatible.hlsli"
#include "../../include/common/indirect_draw.hlsli"

layout(binding = 0, set = 1, scalar) readonly buffer _DrawIndexedIndirectBuffers { DrawIndexedIndirectEX indirect_draws[]; };


layout(location = 0) out vec2 uv;
layout(location = 1) out flat uint matID;

void main() {
    uint draw_idx = gl_DrawID;
    uint geometry_idx = indirect_draws[draw_idx].geometryID;
    InterleavedVertex vertex = fetchVertex();
    GeometryInfo geometry = geometryInfos[geometry_idx];
    // compute position, uv, matID
    mat4 o2w = ObjectToWorld(geometry);
    vec4 positionWorld =  o2w * vec4(vertex.position, 1);
    gl_Position = globalUniform.cameraData.viewProjMat * positionWorld;
    uv = vertex.texCoords;
    matID = geometry.materialID;
}