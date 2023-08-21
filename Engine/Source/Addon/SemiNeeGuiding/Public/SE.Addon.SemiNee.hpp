#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::SemiNEE {
SE_EXPORT struct InitialSamplePass : public RDG::RayTracingPass {
  InitialSamplePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;

  bool spawnVPL = true;
};

SE_EXPORT struct LeafEncodePass : public RDG::ComputePass {
  LeafEncodePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct TreeInitPass : public RDG::ComputePass {
  TreeInitPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct TreeLeavesPass : public RDG::ComputePass {
  TreeLeavesPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct TreeInternalPass : public RDG::ComputePass {
  TreeInternalPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct TreeMergePass : public RDG::ComputePass {
  TreeMergePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  struct PushConstant {
    Math::vec4 gApproxParams = {0.01, 0.99, 0, 0};
  } pConst;
};

SE_EXPORT struct TileBasedDistPass : public RDG::ComputePass {
  TileBasedDistPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct TileDistExchangePass : public RDG::ComputePass {
  TileDistExchangePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool useExhange = false;
};

SE_EXPORT struct TileDistSamplePass : public RDG::RayTracingPass {
  TileDistSamplePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  int sample_hint = 0;
};

SE_EXPORT struct TileDistPerPixelVisPass : public RDG::RayTracingPass {
  TileDistPerPixelVisPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override;
  int sample_hint = 0;
  Math::ivec2 debugPixel = Math::ivec2(0);
};

SE_EXPORT struct TestQuadSamplePass : public RDG::ComputePass {
  TestQuadSamplePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override;
  struct PushConstant {
    Math::uvec2 resolution = {1280, 720};
    uint32_t sample_batch = 0;
    uint32_t padding;
    Math::vec3 emitterPos = Math::vec3{0.f};
    uint32_t seperator = 0;
    Math::vec3 emitterDirection = Math::vec3{0.f, 0.f, 1.f};
    int sampleUsage = 0;
  } pConst;
};

SE_EXPORT struct DVPLVisualizePass : public RDG::RenderPass {
  DVPLVisualizePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;

  struct PushConstantBuffer {
    int vplIdOffset;
    float gVPLRenderScale = 0.05f;
    float gVPLColorScale = 1.f;
  } pConst;
};

SE_EXPORT struct TileDistVisualizePass : public RDG::RenderPass {
  TileDistVisualizePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override;
  struct PushConstantBuffer {
    Math::ivec2 selectedPixel = Math::ivec2{0, 0};
    float gVPLRenderScale = 2.0f;
    float gVPLColorScale = 1.f;
  } pConst;
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

SE_EXPORT struct GroundTruthPass : public RDG::RayTracingPass {
  GroundTruthPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  int showMode = 0;
};
}