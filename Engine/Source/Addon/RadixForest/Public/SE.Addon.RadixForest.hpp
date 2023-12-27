#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::PrefixSum {
SE_EXPORT struct FloatDummyPass : public RDG::DummyPass {
  FloatDummyPass() { RDG::Pass::init(); }
  virtual auto reflect() noexcept -> RDG::PassReflection override;
};

SE_EXPORT struct SinglePassPrefixSumFloatInitPass : public RDG::ComputePass {
  SinglePassPrefixSumFloatInitPass(int max_num);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void override;
  int max_num = 0;
};

SE_EXPORT struct SinglePassPrefixSumFloatMainPass : public RDG::ComputePass {
  SinglePassPrefixSumFloatMainPass(int max_num);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void override;
  int max_num = 0;
};
}

namespace SIByL::Addon::RadixForest {
SE_EXPORT struct PmfNormalizePass : public RDG::ComputePass {
  PmfNormalizePass(int max_num);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void override;
  int max_num = 0;
};

SE_EXPORT struct RadixForestClearPass : public RDG::ComputePass {
  RadixForestClearPass(int max_num, int partition_num);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void override;
  int max_num = 0;
  int partition_num = 0;
};

SE_EXPORT struct RadixForestConstrPass : public RDG::ComputePass {
  RadixForestConstrPass(int max_num, int partition_num);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void override;
  int max_num = 0;
  int partition_num = 0;
};

SE_EXPORT struct RadixForestBuildGraph : public RDG::Graph {
  RadixForestBuildGraph(int max_num, int partition_num);
};
}