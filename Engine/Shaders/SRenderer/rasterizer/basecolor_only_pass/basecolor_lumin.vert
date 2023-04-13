#version 460
#extension GL_GOOGLE_include_directive : enable

struct PushConstants { uint geometry_idx; };
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

#include "../include/common_vert.h"

layout(location = 0) out vec2 uv;
layout(location = 1) out uint matID;
layout(location = 2) out vec3 normalWS;
layout(location = 3) out vec4 tangentWS;
layout(location = 4) out vec3 posVS;

void main() {
    InterleavedVertex vertex = fetchVertex();
    GeometryInfo geometry = geometryInfos[pushConstants.geometry_idx];
    // compute position, uv, matID
    mat4 o2w = ObjectToWorld(geometry);
    vec4 positionWorld =  o2w * vec4(vertex.position, 1);
    posVS = (globalUniform.cameraData.viewMat * positionWorld).xyz;
    gl_Position = globalUniform.cameraData.viewProjMat * positionWorld;
    uv = vertex.texCoords;
    matID = geometry.materialID;
    // compute normal, tangent
    normalWS = normalize((ObjectToWorldNormal(geometry) * vec4(vertex.normal, 0)).xyz);
    tangentWS = vec4(normalize((o2w * vec4(vertex.tangent, 0)).xyz), geometry.oddNegativeScaling);
}