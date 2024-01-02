#pragma once
#include <SE.RDG.hpp>

namespace SIByL {
SE_EXPORT struct SkinningPass : public RDG::ComputePass {
  SkinningPass();
  virtual auto reflect() noexcept -> RDG::PassReflection;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void;
  std::vector<RHI::BindGroupEntry> entries;
  uint32_t vertex_max = 0; uint32_t joint_max = 0;
};
}  // namespace SIByL