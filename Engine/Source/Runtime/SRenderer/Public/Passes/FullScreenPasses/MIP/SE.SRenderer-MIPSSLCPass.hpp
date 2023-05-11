#pragma once
#include <imgui.h>
#include <imgui_internal.h>
#include <array>
#include <compare>
#include <filesystem>
#include <functional>
#include <memory>
#include <typeinfo>
#include <vector>
#include "../../../../../Application/Public/SE.Application.Config.h"
#include "../SE.SRenderer-Blit.hpp"
#include "../../../SE.SRenderer.hpp"

namespace SIByL {
SE_EXPORT struct MIPSSLCInitSubPass : public RDG::FullScreenPass {
  MIPSSLCInitSubPass();
  auto reflect() noexcept -> RDG::PassReflection;

  struct PushConstant {
    Math::mat4 trans_inv_view;
    Math::mat4 inv_proj;
    Math::ivec2 resolution;
    int32_t importance_operator;
    uint32_t modulateJacobian;
  } pConst = {};

  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;

  Core::GUID frag;
};

SE_EXPORT struct MIPSSLCSubPass : public RDG::FullScreenPass {
  MIPSSLCSubPass(size_t mip_offset);

  virtual auto reflect() noexcept -> RDG::PassReflection;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;

  size_t mipOffset;
  Core::GUID frag;
};

SE_EXPORT struct MIPSSLCPass : public RDG::Subgraph {
  MIPSSLCPass(size_t width, size_t height);

  virtual auto alias() noexcept -> RDG::AliasDict override;
  virtual auto onRegister(RDG::Graph* graph) noexcept -> void override;

  size_t mipCount;
  size_t width, height;
};

SE_EXPORT struct MISCompensationDiffPass : public RDG::ComputePass {
  MISCompensationDiffPass();
  virtual auto reflect() noexcept -> RDG::PassReflection;
  struct PushConstant {
    Math::uvec2 resolution;
    float diffWeight = 0.f;
  };
  PushConstant pConst;

  virtual auto renderUI() noexcept -> void override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;

  Core::GUID comp;
};
}  // namespace SIByL