#pragma once
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.RHI.hpp>
#include "../Passes/RasterizerPasses/SE.SRenderer-AlbedoPass.hpp"
#include "../Passes/RasterizerPasses/SE.SRenderer-PreZPass.hpp"
#include "../Passes/RasterizerPasses/SE.SRenderer-FooPass.hpp"
#include "../Passes/RasterizerPasses/SE.SRenderer-ShadowmapPass.hpp"
#include "../Passes/RasterizerPasses/SE.SRenderer-ForwardPass.hpp"
#include "../Passes/RasterizerPasses/SE.SRenderer-CascadeShadowmapPass.hpp"
#include "../Passes/RasterizerPasses/SE.SRenderer-GBufferPass.hpp"
#include "../Passes/FullScreenPasses/SE.SRenderer-ACEsPass.hpp"
#include "../Passes/FullScreenPasses/SE.SRenderer-AccumulatePass.hpp"
#include "../Passes/FullScreenPasses/MIP/SE.SRenderer-MIPMinPoolingPass.hpp"
#include "../Passes/FullScreenPasses/MIP/SE.SRenderer-MIPSSLCPass.hpp"
#include "../Passes/FullScreenPasses/MIP/SE.SRenderer-MIPAddPoolingPass.hpp"
#include "../Passes/FullScreenPasses/ScreenSpace/SE.SRenderer-BarPass.hpp"
#include "../Passes/FullScreenPasses/ScreenSpace/SE.SRenderer-BarLCPass.hpp"
#include "../Passes/FullScreenPasses/ScreenSpace/SE.SRenderer-RSSGeoReconstrPass.hpp"
#include "../Passes/RayTracingPasses/SE.SRenderer-SSRXGTPass.hpp"

namespace SIByL::SRP {
SE_EXPORT struct SSRXForwardGraph : public RDG::Graph {
  SSRXForwardGraph();
};

SE_EXPORT struct SSRXGeoReconstrGraph : public RDG::Graph {
  SSRXGeoReconstrGraph() {
    addPass(std::make_unique<RSSGeoReconstrPass>(512, 512), "Reconstr Pass");
    markOutput("Reconstr Pass", "Depth");

    reconstrPass = static_cast<RSSGeoReconstrPass*>(getPass("Reconstr Pass"));
  }

  RSSGeoReconstrPass* reconstrPass = nullptr;
};

SE_EXPORT struct SSRXGeoRenderGraph : public RDG::Graph {
  SSRXGeoRenderGraph() {
    addPass(std::make_unique<SSRXGTPass>(), "SSRXGT Pass");
    addPass(std::make_unique<AccumulatePass>(), "Accumulate Pass");

    addEdge("SSRXGT Pass", "Color", "Accumulate Pass", "Input");

    markOutput("Accumulate Pass", "Output");

    gtPass = static_cast<SSRXGTPass*>(getPass("SSRXGT Pass"));
  }
  SSRXGTPass* gtPass = nullptr;
};

SE_EXPORT struct SSRXPipeline : public RDG::Pipeline {

  enum struct State {
    Forward,
    GeoReconstr,
    GeoRender,
  } state = State::Forward;

  SSRXPipeline() = default;
  virtual auto build() noexcept -> void;
  virtual auto renderUI() noexcept -> void;
  virtual auto execute(RHI::CommandEncoder* encoder) noexcept -> void;
  virtual auto getActiveGraphs() noexcept -> std::vector<RDG::Graph*>;
  virtual auto getOutput() noexcept -> GFX::Texture*;

  std::unique_ptr<RHI::BLAS> blas = nullptr;
  std::unique_ptr<RHI::TLAS> tlas = nullptr;
  std::unique_ptr<RHI::BLAS> blas_back = nullptr;
  std::unique_ptr<RHI::TLAS> tlas_back = nullptr;

  GFX::Buffer* vertex_buffer = nullptr;
  GFX::Buffer* index_buffer = nullptr;
  bool to_build = false;

  SSRXForwardGraph      forwardGraph;
  SSRXGeoReconstrGraph  geoReconstrGraph;
  SSRXGeoRenderGraph    geoRenderGraph;
};
}