#pragma once
#include <SE.Core.Utility.hpp>
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::LTC {
struct LTCCommon {
  SINGLETON(LTCCommon, ;);  // Private ctor only used by
  GFX::Texture* HorizonClipLUT;	// Lut for sphere horizon-clipping
  GFX::Sampler* lutSampler;
};
}