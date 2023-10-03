#pragma once
#include <SE.Addon.BitonicSort.hpp>
#include <SE.Addon.VPL.hpp>
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::SST {
SE_EXPORT struct TreeEncodePass : public RDG::ComputePass {
  TreeEncodePass(VPL::VPLSpawnInfo* info);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VPL::VPLSpawnInfo* vplInfo;
};

SE_EXPORT struct TreeInitPass : public RDG::ComputePass {
  TreeInitPass(VPL::VPLSpawnInfo* info);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VPL::VPLSpawnInfo* vplInfo;
};

SE_EXPORT struct TreeAssignLeafIndex : public RDG::ComputePass {
  TreeAssignLeafIndex(VPL::VPLSpawnInfo* info);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VPL::VPLSpawnInfo* vplInfo;
};

SE_EXPORT struct TreeInternalNodes : public RDG::ComputePass {
  TreeInternalNodes(VPL::VPLSpawnInfo* info);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  VPL::VPLSpawnInfo* vplInfo;
};

SE_EXPORT struct TreeMergeNodes : public RDG::ComputePass {
  TreeMergeNodes(VPL::VPLSpawnInfo* info);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VPL::VPLSpawnInfo* vplInfo;
  struct ApproxParams {
    float minNormalScore = 0.25f;
    float maxNormalZStd = 0.1f;
  } params;
};

SE_EXPORT struct TreeVisualizePass : public RDG::RenderPass {
  TreeVisualizePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;

  struct PushConstantBuffer {
    int vplIdOffset;
    uint32_t packedSetting = 0;
    float gVPLRenderScale = 2.0f;
    float gVPLColorScale = 1.f;
  } pConst;
};

SE_EXPORT struct SSTGIPass : public RDG::RayTracingPass {
  SSTGIPass(VPL::VPLSpawnInfo* info);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  VPL::VPLSpawnInfo* info;
  int spp = 1;
};

SE_EXPORT struct SSTTestGraph : public RDG::Graph {
  SSTTestGraph();
  VPL::VPLSpawnInfo spawn_info;
  BitonicSort::BitonicSortSetting sort_info;
};

SE_EXPORT struct SSTTestPipeline : public RDG::SingleGraphPipeline {
  SSTTestPipeline() { pGraph = &graph; }
  SSTTestGraph graph;
};
}