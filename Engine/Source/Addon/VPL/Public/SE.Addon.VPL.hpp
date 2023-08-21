#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::VPL {
SE_EXPORT struct VPLSpawnInfo {
  uint32_t maxDepth;
  uint32_t maxNumber;
  bool respawn = true;
};

SE_EXPORT struct DVPLPack {
  GFX::Buffer* pb;  // position buffer
  GFX::Buffer* nb;  // normal buffer
  GFX::Buffer* cb;  // color buffer
  GFX::Buffer* counter; // counter buffer

  static auto fetchPack(RDG::RenderData const& renderData) noexcept -> DVPLPack;
  static auto addEdge(std::string const& src, std::string const& tgt,
                      RDG::Graph* graph) noexcept -> void;
  auto bindPack(RDG::PipelinePass* pass, RDG::RenderContext* context) noexcept
      -> void;
};

SE_EXPORT struct CounterInvalidPass : public RDG::DummyPass {
  CounterInvalidPass(VPLSpawnInfo* info);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VPLSpawnInfo* info;
};

SE_EXPORT struct VPLSpawnPass : public RDG::RayTracingPass {
  VPLSpawnPass(VPLSpawnInfo* info);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VPLSpawnInfo* info;
};

SE_EXPORT struct VPLVisualizePass : public RDG::RenderPass {
  VPLVisualizePass(VPLSpawnInfo* info);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;

  struct PushConstantBuffer {
    float gVPLRenderScale = 0.05f;
    float gVPLColorScale = 1.f;
  } pConst;

  VPLSpawnInfo* info;
};

SE_EXPORT struct VPLTestGraph : public RDG::Graph {
  VPLTestGraph();
  VPLSpawnInfo spawn_info;
};

SE_EXPORT struct VPLTestPipeline : public RDG::SingleGraphPipeline {
  VPLTestPipeline() { pGraph = &graph; }
  VPLTestGraph graph;
};
}  // namespace SIByL::Addon::VPL