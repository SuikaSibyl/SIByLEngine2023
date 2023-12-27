#pragma once
#include <SE.SRenderer.hpp>
#include <SE.Addon.RestirGI.hpp>

namespace SIByL::Addon::SSGuiding {
SE_EXPORT struct SSPGvMF_ClearPass : public RDG::ComputePass {
  SSPGvMF_ClearPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool clear = true;
};

SE_EXPORT struct SSPGvMF_SamplePass : public RDG::RayTracingPass {
  SSPGvMF_SamplePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  int strategy = 0;
  bool learn = true;
  bool multi_bounce = false;
  bool extra_half_spp = false;
  bool learn_one_frame = false;
  float expon_factor = 0.7f;
  int spp = 1;
  bool learn_first = false;
};

SE_EXPORT struct SSPGvMF_SampleReSTIRPass : public RDG::RayTracingPass {
  SSPGvMF_SampleReSTIRPass(RestirGI::GIResamplingRuntimeParameters* param);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  int strategy = 0;
  bool learn = true;
  bool multi_bounce = false;
  bool extra_half_spp = false;
  bool learn_one_frame = false;
  float expon_factor = 0.7f;
  int spp = 1;
  bool learn_first = false;
  RestirGI::GIResamplingRuntimeParameters* param;
};

SE_EXPORT struct SSPGvMF_LearnPass : public RDG::ComputePass {
  SSPGvMF_LearnPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool learn = true;
  bool learn_one_frame = false;
  int extra_sample = 10;
};

SE_EXPORT struct SSPGvMF_VisPass : public RDG::RayTracingPass {
  SSPGvMF_VisPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override;
  int debugMode = 0;
  float scalar = 1.f;
  Math::ivec2 debugPixel = {0, 0};
};

namespace RealGMM {
SE_EXPORT struct SSPGGMM_ClearPass : public RDG::ComputePass {
  SSPGGMM_ClearPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool clear = true;
};

SE_EXPORT struct SSPGGMM_SamplePass : public RDG::RayTracingPass {
  SSPGGMM_SamplePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  int strategy = 0;
  bool learn = true;
  bool multi_bounce = false;
  bool extra_half_spp = false;
  bool learn_one_frame = false;
  float expon_factor = 0.7f;
  int spp = 1;
  bool learn_first = false;
};

SE_EXPORT struct SSPGGMM_LearnPass : public RDG::ComputePass {
  SSPGGMM_LearnPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool learn = true;
  bool learn_one_frame = false;
  int extra_sample = 10;
};

SE_EXPORT struct SSPGGMM_CopyPass : public RDG::ComputePass {
  SSPGGMM_CopyPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void override;
};

SE_EXPORT struct SSPGGMM_VisPass : public RDG::RayTracingPass {
  SSPGGMM_VisPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override;
  int debugMode = 0;
  float scalar = 1.f;
  Math::ivec2 debugPixel = {0, 0};
};
}

SE_EXPORT struct SSPGGMM_ClearPass : public RDG::ComputePass {
  SSPGGMM_ClearPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool clear = true;
};

SE_EXPORT struct SSPGGMM_SamplePass : public RDG::RayTracingPass {
  SSPGGMM_SamplePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  int strategy = 0;
  bool learn = true;
  bool multi_bounce = false;
  bool extra_half_spp = false;
  bool learn_one_frame = false;
  float expon_factor = 0.7f;
  int spp = 1;
  bool learn_first = false;
};

SE_EXPORT struct SSPGGMM_LearnPass : public RDG::ComputePass {
  SSPGGMM_LearnPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool learn = true;
  bool learn_one_frame = false;
  int extra_sample = 10;
};

SE_EXPORT struct SSPGGMM_VisPass : public RDG::RayTracingPass {
  SSPGGMM_VisPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override;
  int debugMode = 0;
  float scalar = 1.f;
  Math::ivec2 debugPixel = {0, 0};
};

SE_EXPORT struct SSPGGMM_TestPass : public RDG::ComputePass {
  SSPGGMM_TestPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override;
  Math::ivec2 debugPixel = {0, 0};
};

SE_EXPORT struct PdfAccum_ClearPass : public RDG::ComputePass {
  PdfAccum_ClearPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool clear = true;
};

SE_EXPORT struct PdfAccum_ViewerPass : public RDG::ComputePass {
  PdfAccum_ViewerPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  float scalar = 1.f;
};

SE_EXPORT struct PdfNormalize_ClearPass : public RDG::ComputePass {
  PdfNormalize_ClearPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool clear = true;
};

SE_EXPORT struct PdfNormalize_SumPass : public RDG::ComputePass {
  PdfNormalize_SumPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct PdfNormalize_ViewerPass : public RDG::ComputePass {
  PdfNormalize_ViewerPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  float scalar = 1.f;
};

SE_EXPORT struct PdfNormalize_TestPass : public RDG::RayTracingPass {
  PdfNormalize_TestPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override;
  Math::ivec2 debugPixel = {0, 0};
};

SE_EXPORT struct CDQ_PresamplePass : public RDG::RayTracingPass {
  CDQ_PresamplePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override;
  Math::ivec2 debugPixel = {0, 0};
};

SE_EXPORT struct CDQ_AdaptionPass : public RDG::ComputePass {
  CDQ_AdaptionPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool initialize = true;
  bool learn = false;
  bool learn_one_frame = false;
};

SE_EXPORT struct CDQ_VisualizePass : public RDG::ComputePass {
  CDQ_VisualizePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};
}