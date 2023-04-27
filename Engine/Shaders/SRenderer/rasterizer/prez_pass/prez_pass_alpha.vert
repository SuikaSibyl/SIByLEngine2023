#version 460
#extension GL_GOOGLE_include_directive : enable

struct PushConstants {
    uint geometry_idx;
    uint rand_seed;
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

#include "../include/common_vert.h"

layout(location = 0) out vec2 uv;
layout(location = 1) out flat uint matID;

void main() {
    InterleavedVertex vertex = fetchVertex();
    GeometryInfo geometry = geometryInfos[pushConstants.geometry_idx];
    // compute position, uv, matID
    mat4 o2w = ObjectToWorld(geometry);
    vec4 positionWorld =  o2w * vec4(vertex.position, 1);
    gl_Position = globalUniform.cameraData.viewProjMat * positionWorld;
    uv = vertex.texCoords;
    matID = geometry.materialID;
}