#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::Fluid {
SE_EXPORT struct LBMD2Q9Pass : public RDG::ComputePass {
  LBMD2Q9Pass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  virtual auto renderUI() noexcept -> void override;
};

SE_EXPORT struct LBMVisPass : public RDG::ComputePass {
  LBMVisPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
};

SE_EXPORT struct LBMGraph : public RDG::Graph {
  LBMGraph();
};

SE_EXPORT struct LBMPipeline : public RDG::SingleGraphPipeline {
  LBMPipeline() { pGraph = &graph; }
  LBMGraph graph;
};
}