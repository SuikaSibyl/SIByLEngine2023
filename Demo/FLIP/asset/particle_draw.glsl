#ifndef _common_h_
#define _common_h_

#include "camera_def.glsl"

struct PushConstants { 
    float scale;
    float padding0;
    float padding1;
    float padding2;
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };
layout(binding = 0, set = 0, scalar) uniform _GlobalUniforms  { CameraData gCamera; };
layout(binding = 1, set = 0, scalar) buffer  _PositionBuffer  { vec2 positions[]; };
layout(binding = 2, set = 0, scalar) buffer  _ParticleColBuffer  { vec3 colors[]; };

#endif // _common_h_