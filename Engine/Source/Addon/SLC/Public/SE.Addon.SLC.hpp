#pragma once
#include <SE.SRenderer.hpp>
#include <SE.Addon.VPL.hpp>
#include <SE.Addon.BitonicSort.hpp>

namespace SIByL::Addon::SLC {
SE_EXPORT struct MortonCodePass : public RDG::ComputePass {
  MortonCodePass(VPL::VPLSpawnInfo* info);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VPL::VPLSpawnInfo* vplInfo;
};

SE_EXPORT struct GenLevel0Pass : public RDG::ComputePass {
  GenLevel0Pass(VPL::VPLSpawnInfo* info);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VPL::VPLSpawnInfo* vplInfo;
  uint32_t numTreeLevels;
  uint32_t numTreeLights;
  uint32_t numStorageNodes;
};

SE_EXPORT struct GenLevelInternalPass : public RDG::ComputePass {
  struct ParameterSet {
    int srcLevel;
    int dstLevelStart;
    int dstLevelEnd;
    int numLevels;
    int numDstLevelsLights;
  };
  GenLevelInternalPass(ParameterSet const& para);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  ParameterSet para;
};

SE_EXPORT struct SLCVisualizePass : public RDG::RenderPass {
  SLCVisualizePass(VPL::VPLSpawnInfo* info);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;

  struct PushConstantBuffer {
    Math::vec2 resolution;
    float line_width = 5.f;
    int showLevel = -1;
  } pConst;

  VPL::VPLSpawnInfo* info;
  int numTreeLights;
};

SE_EXPORT struct SLCGIPass : public RDG::RayTracingPass {
  SLCGIPass(VPL::VPLSpawnInfo* info);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VPL::VPLSpawnInfo* info;
  int leafStartIndex;
  bool useApproximateGoemetry = true;
  bool useLightCone = false;
  int distanceType = 0;
};

SE_EXPORT struct SLCBuildGraph : public RDG::Subgraph {
  SLCBuildGraph(VPL::VPLSpawnInfo* spawn_info);
  virtual auto alias() noexcept -> RDG::AliasDict override;
  virtual auto onRegister(RDG::Graph* graph) noexcept -> void override;
  VPL::VPLSpawnInfo* spawn_info;
  BitonicSort::BitonicSortSetting sort_info;
};

SE_EXPORT struct SLCTestGraph : public RDG::Graph {
  SLCTestGraph();
  VPL::VPLSpawnInfo spawn_info;
  BitonicSort::BitonicSortSetting sort_info;
};

SE_EXPORT struct SLCTestPipeline : public RDG::SingleGraphPipeline {
  SLCTestPipeline() { pGraph = &graph; }
  SLCTestGraph graph;
};
}