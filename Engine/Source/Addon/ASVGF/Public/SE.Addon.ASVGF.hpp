#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::ASVGF {
SE_EXPORT struct Prelude : public RDG::DummyPass {
  Prelude();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
};

SE_EXPORT struct GradientReprojection : public RDG::ComputePass {
  GradientReprojection();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  bool init_rand = true;
};

SE_EXPORT struct GradientImagePass : public RDG::ComputePass {
  GradientImagePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct GradientAtrousPass : public RDG::ComputePass {
  GradientAtrousPass(int iter);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  const int iteration;
};

SE_EXPORT struct TemporalPass : public RDG::ComputePass {
  TemporalPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct AtrousPass : public RDG::ComputePass {
  AtrousPass(int iter);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  const int iteration;
};

SE_EXPORT struct DebugViewer : public RDG::ComputePass {
  DebugViewer();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct FinaleGraph : public RDG::Subgraph {
  virtual auto alias() noexcept -> RDG::AliasDict override;
  virtual auto onRegister(RDG::Graph* graph) noexcept -> void override;
};
}