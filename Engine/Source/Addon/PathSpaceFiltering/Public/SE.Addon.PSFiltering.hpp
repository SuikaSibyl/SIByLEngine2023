#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::PSFiltering {
SE_EXPORT struct HashFunctionViewerPass : public RDG::FullScreenPass {
  HashFunctionViewerPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;
  int flag = 0;
};

namespace Pipeline {
SE_EXPORT struct HashViewPipeline : public RDG::SingleGraphPipeline {
  SE_EXPORT struct HashViewGraph : public RDG::Graph {
    HashViewGraph() {
      addPass(std::make_unique<HashFunctionViewerPass>(), "HashViewer Pass");
      markOutput("HashViewer Pass", "Color"); } };
  HashViewPipeline() { pGraph = &graph; }
  HashViewGraph graph;
};
}
}