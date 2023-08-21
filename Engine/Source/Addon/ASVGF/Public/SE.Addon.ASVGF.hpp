#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::ASVGF {
SE_EXPORT struct Prelude : public RDG::DummyPass {

};

SE_EXPORT struct GradientReprojection : public RDG::ComputePass {
  GradientReprojection();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

}