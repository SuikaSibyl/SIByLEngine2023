#pragma once
#include <Resource/SE.Core.Resource.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.RHI.hpp>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

#include "../Passes/RasterizerPasses/SE.SRenderer-PreZPass.hpp"
#include "../Passes/RasterizerPasses/SE.SRenderer-GeometryInspectorPass.hpp"

namespace SIByL::SRP {
SE_EXPORT struct GeoInspectGraph : public RDG::Graph {
  GeoInspectGraph() {
    addSubgraph(std::make_unique<PreZPass>(), "Pre-Z Pass");
    addPass(std::make_unique<GeometryInspectorPass>(), "GeoInspect Pass");
    addEdge("Pre-Z Pass", "Depth", "GeoInspect Pass", "Depth");
    markOutput("GeoInspect Pass", "Color");
  }
};

SE_EXPORT struct GeoInspectPipeline : public RDG::SingleGraphPipeline {
  GeoInspectPipeline() { pGraph = &graph; }
  GeoInspectGraph graph;
};
}  // namespace SIByL::SRP