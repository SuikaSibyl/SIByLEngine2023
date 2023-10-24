#pragma once
#include <SE.SRenderer.hpp>

namespace SIByL::Addon::Differentiable {
SE_EXPORT enum struct DiffResourceType {
  DIFF_RESOURCE_TYPE_NONE = 0,
  DIFF_RESOURCE_TYPE_TEXTURE = 1,
  DIFF_RESOURCE_TYPE_BUFFER = 2,
};

SE_EXPORT struct DiffResourceDesc {
  uint32_t data_size;    // the size of the data in bytes
  uint32_t data_offset;  // the offset of the data in bytes
  uint32_t data_pitch;   // texture width in pixels
  uint32_t data_flag;    // mask (4 bits) | type (2 bits)
};

SE_EXPORT struct DifferentiableConfigure {
  float learning_rate = 0.01f;
  std::string gt_filepath = "";
  GFX::Texture* ground_truth = nullptr;
  size_t gradient_buffer_size = 0;
  size_t gradient_buffer_elements = 0;
  std::vector<int> texture_indices_host;
  std::unique_ptr<RHI::Buffer> texture_indices_device;
  std::vector<DiffResourceDesc> diff_resources_host;
  std::vector<GFX::Texture*> diff_textures_host;
  std::unique_ptr<RHI::Buffer> diff_resources_device;
  std::unique_ptr<RHI::Buffer> clean_resources_device;
};

SE_EXPORT struct TestGTPass : public RDG::RayTracingPass {
  TestGTPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct TestADPass : public RDG::RayTracingPass {
  TestADPass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
  virtual auto renderUI() noexcept -> void override;
  bool learn = false;
  bool initialize = true;
  float learning_rate = 0.01f;
  GFX::Texture* gt = nullptr;
};

SE_EXPORT struct GradientClearPass : public RDG::ComputePass {
  GradientClearPass(DifferentiableConfigure* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  DifferentiableConfigure* config;
};

SE_EXPORT struct GradientDescentTexPass : public RDG::ComputePass {
  GradientDescentTexPass(DifferentiableConfigure* config, GFX::Texture* texture, uint32_t rid);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  DifferentiableConfigure* config;
  GFX::Texture* texture = nullptr;
  uint32_t rid = 0;
};

SE_EXPORT struct ForwardReferencePass : public RDG::DummyPass {
  ForwardReferencePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
};
// Adjoint Pass
SE_EXPORT struct AdjointRenderPass : public RDG::ComputePass {
  AdjointRenderPass(DifferentiableConfigure* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  virtual auto renderUI() noexcept -> void override;
  DifferentiableConfigure* config;
  int loss_func = 0;
};
// Radiative Backpropagation Pass
SE_EXPORT struct RadiativeBackpropPass : public RDG::RayTracingPass {
  RadiativeBackpropPass(DifferentiableConfigure* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  virtual auto renderUI() noexcept -> void override;
  DifferentiableConfigure* config;
  int max_depth = 1;
};

SE_EXPORT struct AutoDiffGraph : public RDG::Graph {
  //Addon::Differentiable::ParameterSetting setting;
  AutoDiffGraph(DifferentiableConfigure* configure);
};

SE_EXPORT struct AutoDiffPipeline : public RDG::SingleGraphPipeline {
  AutoDiffPipeline() : graph(&configure) { pGraph = &graph; }
  virtual auto renderUI() noexcept -> void override;
  AutoDiffGraph graph;
  DifferentiableConfigure configure;
};
}