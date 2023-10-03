#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::RestirGI {
SE_EXPORT struct GIResamplingRuntimeParameters {
  uint32_t reservoirArrayPitch;     // number of elements in a whole reservoir array
  uint32_t reservoirBlockRowPitch;  // number of elements in a row of reservoir blocks
  uint32_t uniformRandomNumber;
  uint32_t neighborOffsetMask;
};

SE_EXPORT inline auto InitializeParameters(
    uint32_t width, uint32_t height
) noexcept -> GIResamplingRuntimeParameters {
  GIResamplingRuntimeParameters param;
  uint32_t renderWidthBlocks = (width + 16 - 1) / 16;
  uint32_t renderHeightBlocks = (height + 16 - 1) / 16;
  param.reservoirBlockRowPitch = renderWidthBlocks * (16 * 16);
  param.reservoirArrayPitch = param.reservoirBlockRowPitch * renderHeightBlocks;
  return param;
}

SE_EXPORT struct InitialSample : public RDG::RayTracingPass {
  InitialSample(GIResamplingRuntimeParameters* param);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  GIResamplingRuntimeParameters* param;
  bool extra_bounce = false;
};

SE_EXPORT struct TemporalResampling : public RDG::RayTracingPass {
  TemporalResampling(GIResamplingRuntimeParameters* param);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  GIResamplingRuntimeParameters* param;
};

SE_EXPORT struct SpatialResampling : public RDG::RayTracingPass {
  SpatialResampling(GIResamplingRuntimeParameters* param);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  GIResamplingRuntimeParameters* param;
  static auto initNeighborOffsetBuffer() noexcept -> void;
  static GFX::Buffer* neighbor_buffer;
};

SE_EXPORT struct FinalShading : public RDG::RayTracingPass {
  FinalShading(GIResamplingRuntimeParameters* param);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  GIResamplingRuntimeParameters* param;
  bool re_evaluate_radiance = false;
  bool extra_bounce = false;
};
}