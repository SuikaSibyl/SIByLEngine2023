#pragma once
#include "SE.Addon.Differentiable-common.hpp"

namespace SIByL::Addon::Differentiable {
// Radiative Backpropagation Pass
SE_EXPORT struct WasSimple : public RDG::RayTracingPass {
  WasSimple();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
};
}