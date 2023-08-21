#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::VBuffer {
SE_EXPORT struct RayTraceVBuffer : public RDG::RayTracingPass {
  RayTraceVBuffer();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct VBuffer2GBufferPass : public RDG::ComputePass {
  VBuffer2GBufferPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct TestVBuffer : public RDG::RayTracingPass {
  TestVBuffer();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};
}