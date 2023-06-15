#include <cstdint>
#include <SE.Math.Geometric.hpp>
#include <SE.Image.hpp>
using namespace SIByL::Math;

#define in
#define out
#define inout

namespace SIByL::HSL {
SE_EXPORT struct sampler2D {
  uint32_t width, height;
};

// retrieve the dimensions of a level of a texture
SE_EXPORT ivec2 textureSize(sampler2D sampler, int lod);

SE_EXPORT vec4 texelFetch(sampler2D sampler, ivec2 P, int lod);
}