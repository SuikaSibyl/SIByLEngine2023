#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon {
SE_EXPORT struct RasterizedGBufferPass : public RDG::RenderPass {
  RasterizedGBufferPass();
  virtual auto reflect() noexcept -> RDG::PassReflection;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;
};

SE_EXPORT struct GBufferInspectorPass : public RDG::FullScreenPass {
  GBufferInspectorPass();
  virtual auto reflect() noexcept -> RDG::PassReflection;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;
  uint32_t showEnum = 0;
};

SE_EXPORT struct GBufferUtils {
  static auto addGBufferInput(RDG::PassReflection& reflector,
                              uint32_t stages) noexcept -> void;
  static auto addGBufferInputOutput(RDG::PassReflection& reflector,
                              uint32_t stages) noexcept -> void;
  static auto addPrevGBufferInput(RDG::PassReflection& reflector,
                                  uint32_t stages) noexcept -> void;
  static auto addPrevGbufferInputOutput(RDG::PassReflection& reflector,
                                        uint32_t stages) noexcept -> void;

  static auto addGBufferEdges(RDG::Graph* graph, std::string const& src,
                              std::string const& dst) noexcept -> void;
  static auto addPrevGBufferEdges(RDG::Graph* graph, std::string const& src,
                                  std::string const& dst) noexcept -> void;
  static auto addBlitPrevGBufferEdges(RDG::Graph* graph, std::string const& src,
                                      std::string const& tgt,
                                      std::string const& dst) noexcept -> void;


  static auto bindGBufferResource(RDG::PipelinePass* pipeline,
                                  RDG::RenderContext* context,
                                  RDG::RenderData const& renderData) noexcept
      -> void;
  static auto bindPrevGBufferResource(
      RDG::PipelinePass* pipeline, RDG::RenderContext* context,
      RDG::RenderData const& renderData) noexcept -> void;
};

SE_EXPORT struct GBufferTemporalInspectorPass : public RDG::FullScreenPass {
  GBufferTemporalInspectorPass();
  virtual auto reflect() noexcept -> RDG::PassReflection;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;
  uint32_t showEnum = 0;
};

SE_EXPORT struct GBufferHolderSource : public RDG::DummyPass {
  GBufferHolderSource();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
};

SE_EXPORT struct GBufferHolderGraph : public RDG::Subgraph {
  virtual auto alias() noexcept -> RDG::AliasDict override;
  virtual auto onRegister(RDG::Graph* graph) noexcept -> void override;
};

SE_EXPORT struct GBufferShading : public RDG::RayTracingPass {
  GBufferShading();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};
}
