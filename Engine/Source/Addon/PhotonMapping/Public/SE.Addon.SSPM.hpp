#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::SSPM {
SE_EXPORT struct SSPMClearPass : public RDG::ComputePass {
  SSPMClearPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct SSPMPass : public RDG::RayTracingPass {
  SSPMPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct SSPMViewPass : public RDG::ComputePass {
  SSPMViewPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct SSPMGraph : public RDG::Graph { SSPMGraph(); };
SE_EXPORT struct SSPMGPipeline : public RDG::SingleGraphPipeline {
  SSPMGPipeline() { pGraph = &graph; } SSPMGraph graph;
};
}