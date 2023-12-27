#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::Lightmap {
SE_EXPORT struct RasterizedLightmapUVPass : public RDG::RenderPass {
  RasterizedLightmapUVPass();
  virtual auto reflect() noexcept -> RDG::PassReflection;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;
};

SE_EXPORT struct LightmapVisualizeGraph : public RDG::Graph {
  LightmapVisualizeGraph();
};

SE_EXPORT struct LightmapVisualizePipeline : public RDG::SingleGraphPipeline {
  LightmapVisualizePipeline() { pGraph = &graph; }
  LightmapVisualizeGraph graph;
};
}  // namespace SIByL::Addon::Lightmap