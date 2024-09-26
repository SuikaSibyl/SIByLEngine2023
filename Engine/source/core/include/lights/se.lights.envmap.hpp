#pragma once
#include <se.rhi.hpp>
#include <se.gfx.hpp>

namespace se {
struct SIByL_API EnvmapLight {
  enum struct ImportanceType {
    Luminance,
    Length,
  };

  ivec2 size;
  vec3 rgb_int;
  gfx::TextureHandle texture;
  gfx::PMFConstructor::PiecewiseConstant2D distribution;

  EnvmapLight(std::string const& path, ImportanceType type = ImportanceType::Luminance);
  auto width() noexcept -> int;
  auto height() noexcept -> int;
  auto rgb_integrated() noexcept -> vec3;
  auto condition_offset() noexcept -> int;
  auto marginal_offset() noexcept -> int;
  auto get_texture() noexcept -> gfx::TextureHandle;
};
}