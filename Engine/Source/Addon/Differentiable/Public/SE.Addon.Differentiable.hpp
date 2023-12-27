#pragma once
#include <SE.SRenderer.hpp>
#include <SE.Addon.RadixForest.hpp>
#include "SE.Addon.Differentiable-common.hpp"
#include "SE.Addon.Differentiable-was.hpp"

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

// Adjoint Pass
SE_EXPORT struct AdjointRenderPass : public RDG::ComputePass {
  AdjointRenderPass(DifferentiableDevice* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  virtual auto renderUI() noexcept -> void override;
  DifferentiableDevice* config;
  int loss_func = 0;
};
// Radiative Backpropagation Pass
SE_EXPORT struct RadiativeBackpropPass : public RDG::RayTracingPass {
  RadiativeBackpropPass(DifferentiableDevice* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  virtual auto renderUI() noexcept -> void override;
  DifferentiableDevice* config;
  int max_depth = 1;
  int spp = 1;
};

// Radiative Backpropagation Pass
SE_EXPORT struct ReparamSimple : public RDG::RayTracingPass {
  ReparamSimple();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  virtual auto renderUI() noexcept -> void override;
};

SE_EXPORT struct ReparamInit : public RDG::ComputePass {
  ReparamInit();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool init = true;
  float learning_rate = 0.00f;
  Math::vec3 initial_value = {0.3f, 0.1f, 0.2f};
};

SE_EXPORT struct AutoDiffGraph : public RDG::Graph {
  AutoDiffGraph() = default;
  AutoDiffGraph(DifferentiableDevice* configure);
  void addModule(IModule* modle) { modules.push_back(modle); }
  std::vector<IModule*> modules;
};

SE_EXPORT struct FusedADPipeline : public RDG::Pipeline {
  FusedADPipeline() = default;
  virtual auto execute(RHI::CommandEncoder* encoder) noexcept -> void override;
  virtual auto getActiveGraphs() noexcept -> std::vector<RDG::Graph*> override;
  virtual auto getOutput() noexcept -> GFX::Texture* override;
  virtual auto readback() noexcept -> void { pGraph->readback(); }
  virtual auto build() noexcept -> void override { pGraph->build(); }
  DifferentiableDevice diffdevice;
  RDG::Graph* pGraph;
};

SE_EXPORT struct AutoDiffPipeline : public FusedADPipeline {
  AutoDiffPipeline();
  virtual auto renderUI() noexcept -> void override;
  AutoDiffGraph graph;
};

SE_EXPORT struct NeuralRadiosityPipeline : public RDG::SingleGraphPipeline {
  NeuralRadiosityPipeline();
  std::unique_ptr<RadixForest::RadixForestBuildGraph> graph;
};
}