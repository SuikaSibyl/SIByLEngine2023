#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::SSGuiding {
SE_EXPORT struct SSPGvMF_ClearPass : public RDG::ComputePass {
  SSPGvMF_ClearPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool clear = true;
};

SE_EXPORT struct SSPGvMF_SamplePass : public RDG::RayTracingPass {
  SSPGvMF_SamplePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  int strategy = 0;
};

SE_EXPORT struct SSPGvMF_VisPass : public RDG::RayTracingPass {
  SSPGvMF_VisPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override;
  int debugMode = 0;
  float scalar = 1.f;
  Math::ivec2 debugPixel = {0, 0};
};
}