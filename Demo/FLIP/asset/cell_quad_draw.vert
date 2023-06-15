#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : require
#include "camera_def.glsl"

layout(binding = 0, set = 0, scalar) uniform _GlobalUniforms  { CameraData gCamera; };
layout(binding = 1, set = 0) uniform sampler2D in_color;

struct PushConstants { 
    vec2 center;
    vec2 diag;
};
layout(push_constant) uniform PushConsts { PushConstants pConst; };

layout(location = 0) out vec2 uv;

vec2 uvs[6] = vec2[](
    // triangle 0
    vec2(0, 0),
    vec2(1, 0),
    vec2(0, 1),
    vec2(1, 0),
    vec2(0, 1),
    vec2(1, 1)
);

void main() {
    uint id = gl_VertexIndex;
	uv = uvs[id];
	const vec3 vertex = vec3(pConst.center + ((uv*2 - vec2(1)) * pConst.diag),-1);
    gl_Position = gCamera.viewProjMat * vec4(vertex, 1);
}