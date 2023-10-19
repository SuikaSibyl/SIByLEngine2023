#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::Differentiable {
SE_EXPORT struct TestGTPass : public RDG::RayTracingPass {
  TestGTPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct TestADPass : public RDG::RayTracingPass {
  TestADPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool learn = false;
  bool initialize = true;
  float learning_rate = 0.01f;
  GFX::Texture* gt = nullptr;
};
}