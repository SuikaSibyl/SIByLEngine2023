#pragma once
#include <se.rdg.hpp>
#include <se.gfx.hpp>

namespace se {
struct SIByL_API RasterizerShadePass : public rdg::RenderPass {
  RasterizerShadePass();
  virtual auto reflect() noexcept -> rdg::PassReflection;
  //virtual auto renderUI() noexcept -> void override;
  virtual auto execute(rdg::RenderContext* context,
    rdg::RenderData const& renderData) noexcept -> void;
  virtual auto beforeDirectDrawcall(rhi::RenderPassEncoder* encoder, int geometry_idx,
    gfx::Scene::GeometryDrawData const& data) noexcept -> void override;
  struct PushConst {
    ivec2 resolution;
    int geometryIndex;
    int cameraIndex;
  } pConst;
};
}