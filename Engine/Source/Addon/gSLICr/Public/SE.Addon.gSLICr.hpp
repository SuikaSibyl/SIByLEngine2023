#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::gSLICr {
SE_EXPORT struct gSLICrSetting {
  Math::ivec2 img_size;
  Math::ivec2 map_size;
  int spixel_size = 32;
  int number_iter = 5;
  bool enforce_connectivity = true;
  float coh_weight = 0.6f;
};

SE_EXPORT struct InitClusterCenterPass : public RDG::ComputePass {
  InitClusterCenterPass(gSLICrSetting* desc);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  gSLICrSetting* desc;
};

SE_EXPORT struct FindCenterAssociationPass : public RDG::ComputePass {
  FindCenterAssociationPass(gSLICrSetting* desc);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  gSLICrSetting* desc;
};

//SE_EXPORT struct UpdateClusterCenterPass : public RDG::ComputePass {
//  UpdateClusterCenterPass(gSLICrSetting const& desc);
//  virtual auto reflect() noexcept -> RDG::PassReflection override;
//  virtual auto execute(RDG::RenderContext* context,
//                       RDG::RenderData const& renderData) noexcept
//      -> void override;
//  gSLICrSetting desc;
//};
//
//SE_EXPORT struct FinalizeReductionResultPass : public RDG::ComputePass {
//  FinalizeReductionResultPass(gSLICrSetting const& desc);
//  virtual auto reflect() noexcept -> RDG::PassReflection override;
//  virtual auto execute(RDG::RenderContext* context,
//                       RDG::RenderData const& renderData) noexcept
//      -> void override;
//  gSLICrSetting desc;
//};
//
//SE_EXPORT struct EnforceConnectivityPass : public RDG::ComputePass {
//  EnforceConnectivityPass(gSLICrSetting const& desc);
//  virtual auto reflect() noexcept -> RDG::PassReflection override;
//  virtual auto execute(RDG::RenderContext* context,
//                       RDG::RenderData const& renderData) noexcept
//      -> void override;
//  gSLICrSetting desc;
//};
//
SE_EXPORT struct VisualizeSPixelPass : public RDG::ComputePass {
  VisualizeSPixelPass(gSLICrSetting const& desc);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  gSLICrSetting desc;
  bool drawBoundary = true;
  int debug_mode = 0;
};

//SE_EXPORT struct gSLICrGraph : public RDG::Subgraph {
//  gSLICrGraph(gSLICrSetting const& desc);
//  virtual auto alias() noexcept -> RDG::AliasDict override;
//  virtual auto onRegister(RDG::Graph* graph) noexcept -> void override;
//  gSLICrSetting desc;
//};
}