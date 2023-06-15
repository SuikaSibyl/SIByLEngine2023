#include "../Public/SE.HSL.hpp"
#include <algorithm>

namespace SIByL::HSL {
ivec2 textureSize(sampler2D sampler, int lod) {
  return ivec2{std::max(int(sampler.width >> lod), 1),
               std::max(int(sampler.height >> lod), 1)};
}

vec4 texelFetch(sampler2D sampler, ivec2 P, int lod) { return vec4{}; }
}