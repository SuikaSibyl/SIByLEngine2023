#pragma once
#include <SE.SRenderer.hpp>
#include <span>
#include <random>

namespace SIByL::Addon::Differentiable {
SE_EXPORT struct TrainingProcedure {
  bool start_training = false;
  bool on_training = false;
  int iteration = 0;
};

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

SE_EXPORT struct ParamInitializer {
  ParamInitializer(int seed = 0);
  auto uniform(std::span<float>, float min = 0, float max = 1) noexcept -> void;
  auto normal(std::span<float>, float mean = 0, float var = 1) noexcept -> void;
  auto constant(std::span<float>, float var = 0) noexcept -> void;
  std::default_random_engine e;
};

SE_EXPORT struct IModule {
  // register the module to the RDG
  virtual auto on_register() noexcept -> void = 0;
  // return the number of params in the module
  virtual auto get_buffer_param_count() noexcept -> uint32_t = 0;
  // return the number of primal in the module
  virtual auto get_primal_buffer_size() noexcept -> uint32_t = 0;
  // initialize the parameters in the modle
  virtual auto initialize_primal(std::span<float> primal,
    ParamInitializer* initializer) noexcept -> void = 0;
  // the offset of the primal and gradient in the buffer
  uint32_t primal_offset = 0;
  uint32_t gradient_offset = 0;
  // the offset of the param in the param buffer
  uint32_t param_offset = 0;
};

SE_EXPORT struct INetworkLayer {
  // return the number of params in the module
  virtual auto get_buffer_param_count() noexcept -> uint32_t = 0;
  // initialize the primal parameters
  virtual auto initialize_primal(std::span<float> primal,
    ParamInitializer* initializer) noexcept -> void = 0;
};

SE_EXPORT struct INetworkModule : public IModule {
    
};

SE_EXPORT struct IInverseRenderModule : public IModule {

};

SE_EXPORT struct GradientResourcePool {

};

SE_EXPORT struct OptimizerInstance {
  // paramter of the optimizer
  struct ParamPack {
    int type_flag = 0;
    int step_count = 0;
    float learn_r = 0.01;
    float param_0 = 0;
    float param_1 = 0;
    float param_2 = 0;
    float param_3 = 0;
    float param_4 = 0;
  } param;
  // buffer for auxillary information
  std::unique_ptr<RHI::Buffer> aux_buffer;
  // get auxiliary multiplier
  auto auxiliary_multiplier() noexcept -> int;
  // render ui for editing the optimizer
  auto renderUI() noexcept -> void;
  // save the status of the optimzier
  auto serialize() noexcept -> void;
  // load the status of the optimizer
  auto deserialize() noexcept -> void;
  // step the optimizer
  auto step(TrainingProcedure* train) noexcept -> void;
};

SE_EXPORT struct DifferentiableDevice {
  float learning_rate = 0.01f;
  bool is_learning = false;
  std::string gt_filepath = "";
  GFX::Texture* ground_truth = nullptr;
  size_t primal_buffer_size = 0;
  size_t gradient_buffer_size = 0;
  size_t gradient_buffer_elements = 0;
  std::vector<int>              texture_indices_host;
  std::unique_ptr<RHI::Buffer>  texture_indices_device;
  std::vector<GFX::Texture*>    diff_textures_host;
  // gradient resource layout descriptor
  std::vector<DiffResourceDesc> diff_resources_host;
  std::unique_ptr<RHI::Buffer>  diff_resources_device;
  // primal buffer resource on host & device
  std::vector<float>            primal_buffer_host;
  std::unique_ptr<RHI::Buffer>  primal_resources_device;
  // indirect ?
  std::unique_ptr<RHI::Buffer> clean_resources_device;
  TrainingProcedure training;
  OptimizerInstance optimizer;
};

SE_EXPORT struct GradientClearPass : public RDG::ComputePass {
  GradientClearPass(DifferentiableDevice* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  DifferentiableDevice* config;
};

SE_EXPORT struct GradientDescentTexPass : public RDG::ComputePass {
  GradientDescentTexPass(DifferentiableDevice* config, GFX::Texture* texture, uint32_t rid);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  DifferentiableDevice* config;
  GFX::Texture* texture = nullptr;
  uint32_t rid = 0;
};

SE_EXPORT struct GradientDescentPrimPass : public RDG::ComputePass {
  GradientDescentPrimPass(DifferentiableDevice* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  DifferentiableDevice* config;
  RHI::Buffer* buffer = nullptr;
};

SE_EXPORT struct LossReadbackTexPass : public RDG::ComputePass {
  LossReadbackTexPass(DifferentiableDevice* config);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  virtual auto renderUI() noexcept -> void override;
  virtual auto readback(RDG::RenderData const& renderData) noexcept
      -> void override;
  float loss = -1.f;
  std::vector<float> loss_history;
  DifferentiableDevice* config = nullptr;
};

SE_EXPORT struct ForwardReferencePass : public RDG::DummyPass {
  ForwardReferencePass();
  virtual auto reflect() noexcept -> RDG::PassReflection override;
};

SE_EXPORT struct LinearLayer : public INetworkLayer {
  LinearLayer(int C = 16); int C;
  virtual auto get_buffer_param_count() noexcept -> uint32_t override;
  virtual auto initialize_primal(std::span<float> primal,
    ParamInitializer* initializer) noexcept -> void override;
};

SE_EXPORT struct MLPLayer : public INetworkLayer {
  MLPLayer(int N = 3, int C = 16); int N; int C;
  virtual auto get_buffer_param_count() noexcept -> uint32_t override;
  virtual auto initialize_primal(std::span<float> primal,
    ParamInitializer* initializer) noexcept -> void override;
};

SE_EXPORT struct FeatureGrid2D : public INetworkLayer {
  FeatureGrid2D(Math::ivec2 grid, int dim); Math::ivec2 grid; int dim;
  virtual auto get_buffer_param_count() noexcept -> uint32_t override;
  virtual auto initialize_primal(std::span<float> primal,
    ParamInitializer* initializer) noexcept -> void override;
};

SE_EXPORT struct MatrixSanityCheck : public RDG::ComputePass, public INetworkModule {
  // ComputePass
  MatrixSanityCheck(DifferentiableDevice* diffd);
  virtual auto reflect() noexcept -> RDG::PassReflection override;
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void override;
  // IModule
  virtual auto on_register() noexcept -> void override;
  virtual auto get_buffer_param_count() noexcept -> uint32_t override;
  virtual auto get_primal_buffer_size() noexcept -> uint32_t override;
  virtual auto initialize_primal(std::span<float> primal,
    ParamInitializer* initializer) noexcept -> void override;
  MLPLayer mlp = MLPLayer(3, 16);
  FeatureGrid2D feature_grid = FeatureGrid2D({32, 32}, 14);
  DifferentiableDevice* diffd = nullptr;
};
}