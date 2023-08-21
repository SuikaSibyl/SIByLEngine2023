#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::BitonicSort {
SE_EXPORT struct BitonicSortSetting {
  enum struct ElementType {
    UINT64,
    UINT32,
  } type = ElementType::UINT64;
  enum struct DispathType {
    DYNAMIC_INDIRECT,
    STATIC_DIRECT,
  } dispath = DispathType::STATIC_DIRECT;
  uint32_t element_count;
  uint32_t element_count_aligned;
  uint32_t counter_offset = 0;
  std::vector<uint32_t> dispatch_args;
  auto generateArgs() noexcept -> void;
};

SE_EXPORT struct IndirectArgsPass : public RDG::ComputePass {
  IndirectArgsPass(BitonicSortSetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  BitonicSortSetting* setting = nullptr;
  uint32_t alignedMaxNumElements;
  uint32_t maxIterations;
};

SE_EXPORT struct PresortPass : public RDG::ComputePass {
  PresortPass(BitonicSortSetting* setting, uint32_t offset = 0);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  BitonicSortSetting* setting = nullptr;
  uint32_t offset;
};

SE_EXPORT struct OuterSortPass : public RDG::ComputePass {
  OuterSortPass(BitonicSortSetting* setting, uint32_t k, uint32_t j,
                uint32_t arg);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  BitonicSortSetting* setting = nullptr;
  uint32_t k;
  uint32_t j;
  uint32_t arg;
};

SE_EXPORT struct InnerSortPass : public RDG::ComputePass {
  InnerSortPass(BitonicSortSetting* setting, uint32_t k, uint32_t arg);
  InnerSortPass(BitonicSortSetting* setting, uint32_t offset);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  BitonicSortSetting* setting = nullptr;
  uint32_t k;
  uint32_t arg;
  uint32_t offset;
};

SE_EXPORT struct BitonicSort : public RDG::Subgraph {
  BitonicSort(BitonicSortSetting* setting);
  virtual auto alias() noexcept -> RDG::AliasDict override;
  virtual auto onRegister(RDG::Graph* graph) noexcept -> void override;
  BitonicSortSetting* setting = nullptr;
  std::string output_pass;
};

SE_EXPORT struct RandomInitPass : public RDG::ComputePass {
  RandomInitPass(BitonicSortSetting* setting);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  BitonicSortSetting* setting = nullptr;
};
}  // namespace SIByL::BitonicSort