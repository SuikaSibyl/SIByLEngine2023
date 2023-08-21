#include "../Public/SE.Addon.LTC.hpp"
#include "SE.Addon.LTC.data.hpp"
#include <SE.Image.hpp>

namespace SIByL::Addon::LTC {
LTCCommon::LTCCommon() {
  Image::Texture_Host hc_lut_host;
  hc_lut_host.extend = {64, 64, 1};
  hc_lut_host.format = RHI::TextureFormat::R32_FLOAT;
  hc_lut_host.dimension = RHI::TextureDimension::TEX2D;
  hc_lut_host.buffer.isReference = true;
  hc_lut_host.buffer.size = sizeof(float) * 64 * 64;
  hc_lut_host.buffer.data = horizonClipMatrix.data();
  hc_lut_host.mip_levels = 1;
  hc_lut_host.array_layers = 1;
  hc_lut_host.data_size = hc_lut_host.buffer.size;
  hc_lut_host.subResources.emplace_back(
      Image::Texture_Host::SubResource{0, 0, 0, hc_lut_host.data_size, 64, 64});
  Core::GUID hc_lut_guid =
      Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
  GFX::GFXManager::get()->registerTextureResource(hc_lut_guid, &hc_lut_host);
  HorizonClipLUT =
      Core::ResourceManager::get()->getResource<GFX::Texture>(hc_lut_guid);

  Core::GUID sampler_guid =
      Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
  GFX::GFXManager::get()->registerSamplerResource(
      sampler_guid,
      RHI::SamplerDescriptor{RHI::AddressMode::CLAMP_TO_EDGE,
                             RHI::AddressMode::CLAMP_TO_EDGE,
                             RHI::AddressMode::CLAMP_TO_EDGE,
                             RHI::FilterMode::LINEAR, RHI::FilterMode::LINEAR});
  lutSampler =
      Core::ResourceManager::get()->getResource<GFX::Sampler>(sampler_guid);
}
}