#pragma once
#include <SE.Math.Geometric.hpp>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>

namespace SIByL {
SE_EXPORT struct BlitPass : public RDG::FullScreenPass {
  struct Descriptor {
    uint32_t src_mip;
    uint32_t src_array;
    uint32_t dst_mip;
    uint32_t dst_array;
  } desc;

  BlitPass(Descriptor const& desc);
  virtual auto reflect() noexcept -> RDG::PassReflection;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;
  Core::GUID vert, frag;
};
}  // namespace SIByL