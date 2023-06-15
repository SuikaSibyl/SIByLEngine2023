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
#include "../../../SE.SRenderer.hpp"
#include "../SE.SRenderer-Blit.hpp"

namespace SIByL {
SE_EXPORT struct MIPSLCSubPass : public RDG::FullScreenPass {
  MIPSLCSubPass(size_t mip_offset);

  virtual auto reflect() noexcept -> RDG::PassReflection;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;

  size_t mipOffset;
  Core::GUID frag;
};

SE_EXPORT struct MIPSLCPass : public RDG::Subgraph {
  MIPSLCPass(size_t width, size_t height);

  virtual auto alias() noexcept -> RDG::AliasDict override;
  virtual auto onRegister(RDG::Graph* graph) noexcept -> void override;

  size_t mipCount;
  size_t width, height;
};

SE_EXPORT struct MIPTiledVisInputPass : public RDG::DummyPass {
  MIPTiledVisInputPass(uint32_t tile_size, uint32_t tile_buffer_size,
                       uint32_t width,
                  uint32_t height);

  virtual auto reflect() noexcept -> RDG::PassReflection;

  uint32_t tile_size, tile_buffer_size, width, height;
};

SE_EXPORT struct MIPTiledVisSubPass : public RDG::FullScreenPass {
  MIPTiledVisSubPass(size_t mip_offset);

  virtual auto reflect() noexcept -> RDG::PassReflection;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;

  size_t mipOffset;
  Core::GUID frag;
};

SE_EXPORT struct MIPTiledVisPass : public RDG::Subgraph {
  MIPTiledVisPass(uint32_t tile_size, uint32_t tile_buffer_size,
                   uint32_t width, uint32_t height);

  virtual auto alias() noexcept -> RDG::AliasDict override;
  virtual auto onRegister(RDG::Graph* graph) noexcept -> void override;

  uint32_t mipCount;
  uint32_t tile_size, tile_buffer_size;
  uint32_t width, height;
};
}  // namespace SIByL