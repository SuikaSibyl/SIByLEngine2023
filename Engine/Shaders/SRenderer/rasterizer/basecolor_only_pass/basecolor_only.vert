#version 460
#extension GL_GOOGLE_include_directive : enable

struct PushConstants { uint geometry_idx; };
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

#include "../../include/common_vert.h"

layout(location = 0) out vec2 uv;
layout(location = 1) out uint matID;

void main() {
    InterleavedVertex vertex = fetchVertex();
    GeometryInfo geometry = geometryInfos[pushConstants.geometry_idx];

    mat4 objTransform = transpose(mat4(geometry.transform[0], geometry.transform[1], geometry.transform[2], vec4(0,0,0,1)));
    vec4 positionWorld =  objTransform * vec4(vertex.position, 1);
    gl_Position = globalUniform.proj * globalUniform.view * positionWorld;
    uv = vertex.texCoords;
    matID = geometry.materialID;
}