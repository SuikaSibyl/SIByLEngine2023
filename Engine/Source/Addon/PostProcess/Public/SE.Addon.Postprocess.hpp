#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::Postprocess {
SE_EXPORT struct AccumulatePass : public RDG::ComputePass {
  AccumulatePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;

  struct PushConstant {
    Math::uvec2 resolution;
    uint32_t gAccumCount;
    uint32_t gAccumulate = 0;
    uint32_t gMovingAverageMode;
  };
  PushConstant pConst;
  int maxAccumCount = 5;
};
}  // namespace SIByL::Addon::Postprocess