#version 460
#extension GL_GOOGLE_include_directive : enable

struct PushConstants { uint geometry_idx; };
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

#include "../include/common_vert.h"

layout(location = 0) out vec2 uv;
layout(location = 1) out uint matID;
layout(location = 2) out vec3 color;
layout(location = 3) out mat3 TBN;

void main() {
    InterleavedVertex vertex = fetchVertex();
    GeometryInfo geometry = geometryInfos[pushConstants.geometry_idx];

    mat4 o2w = ObjectToWorld(geometry);
    vec4 positionWorld =  o2w * vec4(vertex.position, 1);
    gl_Position = globalUniform.proj * globalUniform.view * positionWorld;
    uv = vertex.texCoords;
    matID = geometry.materialID;

    vec3 wNormal = normalize((ObjectToWorldNormal(geometry) * vec4(vertex.normal, 0)).xyz);
    vec3 wTangent = normalize((o2w * vec4(vertex.tangent, 0)).xyz);
    vec3 wBitangent = cross(wNormal, wTangent) * geometry.oddNegativeScaling;
    TBN = mat3(wTangent, wBitangent, wNormal);

    color = vec3(wNormal);
}