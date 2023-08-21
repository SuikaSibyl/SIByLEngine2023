#version 460
#extension GL_GOOGLE_include_directive : enable

#include "../../include/scene_descriptor_sets.h"
#include "../../../Utility/random.h"

struct PushConstants { 
    uint rand_seed;
};
layout(push_constant) uniform PushConsts { PushConstants pushConstants; };

layout(location = 0) in vec2 uv;
layout(location = 1) in flat uint matID;

void main() {
    const uvec2 tid = uvec2(gl_FragCoord.xy);
    uint RNG = InitRNG(tid, pushConstants.rand_seed);

    uint texID = materials[matID].baseOrDiffuseTextureIndex;
    float alpha = texture(textures[texID], uv).w;
    
    float rnd = UniformFloat(RNG);

    if(rnd > alpha) {
        discard;
    }
}