#version 460
#extension GL_GOOGLE_include_directive : enable

struct PushConstants { uint geometry_idx; };
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

#include "../include/common_vert.h"

void main() {
    InterleavedVertex vertex = fetchVertex();
    GeometryInfo geometry = geometryInfos[pushConstants.geometry_idx];
    // compute position, uv, matID
    mat4 o2w = ObjectToWorld(geometry);
    vec4 positionWorld =  o2w * vec4(vertex.position, 1);
    gl_Position = globalUniform.cameraData.viewProjMat * positionWorld;
}