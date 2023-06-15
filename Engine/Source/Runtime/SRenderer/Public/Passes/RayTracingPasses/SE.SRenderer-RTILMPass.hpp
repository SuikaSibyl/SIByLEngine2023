#pragma once

#include <SE.Editor.Core.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.RHI.hpp>
#include <SE.SRenderer.hpp>
#include <bitset>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <typeinfo>

#include "../../../../Application/Public/SE.Application.Config.h"

namespace SIByL::SRP {
SE_EXPORT struct RSMShareInfo {
  Math::mat4 inv_view;
  Math::mat4 inv_proj;
  Math::mat4 proj_view;
  Math::vec3 direction;
  float area;
};

SE_EXPORT struct DirectRSMPass : public RDG::RayTracingPass {
  struct PushConstant {
    Math::mat4 inv_view;
    Math::mat4 inv_proj;
    Math::vec3 direction;
    float padding;
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
  };

  float scaling_x = 1.f;
  float scaling_y = 1.f;

  uint32_t width, height;
  Core::GUID rsm_rgen;
  RSMShareInfo* info = nullptr;

  DirectRSMPass(uint32_t width, uint32_t height, RSMShareInfo* info);

  virtual auto reflect() noexcept -> RDG::PassReflection override;

  virtual auto renderUI() noexcept -> void override;

  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct RSMGIPass : public RDG::RayTracingPass {};
}  // namespace SIByL::SRP