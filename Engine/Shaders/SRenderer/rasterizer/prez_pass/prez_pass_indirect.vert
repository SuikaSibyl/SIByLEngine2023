#version 460
#extension GL_GOOGLE_include_directive : enable

#include "../include/common_vert.h"
#include "../include/common_indirect.h"

layout(binding = 0, set = 1, scalar) readonly buffer _DrawIndexedIndirectBuffers { DrawIndexedIndirectEX indirect_draws[]; };

void main() {
    uint draw_idx = gl_DrawID;
    uint geometry_idx = indirect_draws[draw_idx].geometryID;
    InterleavedVertex vertex = fetchVertex();
    GeometryInfo geometry = geometryInfos[geometry_idx];
    // compute position, uv, matID
    mat4 o2w = ObjectToWorld(geometry);
    vec4 positionWorld =  o2w * vec4(vertex.position, 1);
    gl_Position = globalUniform.cameraData.viewProjMat * positionWorld;
}