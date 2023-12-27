#include "../Public/SE.Addon.Differentiable-common.hpp"
#include <IO/SE.Core.IO.hpp>
#include <cnpy.h>

namespace SIByL::Addon::Differentiable {
enum struct OptimizerType {
    SGD,
    Adam,
};

auto OptimizerInstance::auxiliary_multiplier() noexcept -> int {
  if (param.type_flag == int(OptimizerType::SGD)) { return 1;
  } else if (param.type_flag == int(OptimizerType::Adam)) { return 2;
  } else { return 1; }
}

auto OptimizerInstance::renderUI() noexcept -> void {
  // select type
  int opt_type = param.type_flag;
  const char* kind_names[] = {"SGD", "Adam"};
  ImGui::Combo("Type", &opt_type, kind_names, IM_ARRAYSIZE(kind_names),
               IM_ARRAYSIZE(kind_names));
  bool reset_params = false;
  if (opt_type != param.type_flag) {
    param.type_flag = opt_type;
    reset_params = true;
  }

  // reset parameters
  if (reset_params) {
    param.step_count = 0;
    param.learn_r = 0.01;
    param.param_0 = 0.f;
    param.param_1 = 0.f;
    param.param_2 = 0.f;
    param.param_3 = 0.f;
    param.param_4 = 0.f;
    if (param.type_flag == int(OptimizerType::SGD)) {
    } else if (param.type_flag == int(OptimizerType::Adam)) {
      param.param_0 = 0.9;    // beta_1
      param.param_1 = 0.999;  // beta_2
      param.param_2 = 1e-08;  // epsilon
    }
  }
  // modify parameters
  // common parameters
  ImGui::DragFloat("learning_rate", &param.learn_r, 0.1);
  ImGui::DragInt("step", &param.step_count, 0);
  // specified parameters
  if (param.type_flag == int(OptimizerType::SGD)) {
    ImGui::DragFloat("momentum", &param.param_0, 0.1, 0.f, 1.f);
    ImGui::DragFloat("dampening", &param.param_1, 0.1, 0.f, 1.f);
  } else if (param.type_flag == int(OptimizerType::Adam)) {
    ImGui::DragFloat("beta_1", &param.param_0, 0.1, 0.f, 1.f);
    ImGui::DragFloat("beta_2", &param.param_1, 0.1, 0.f, 1.f);
    ImGui::DragFloat("epsilon", &param.param_2, 0.1, 0.f, 1.f);
  }
}

auto OptimizerInstance::serialize() noexcept -> void {

}

auto OptimizerInstance::deserialize() noexcept -> void {
    
}

auto OptimizerInstance::step(TrainingProcedure* train) noexcept -> void {
  if (train->on_training) {
    param.step_count++;
  }
}

GradientClearPass::GradientClearPass(DifferentiableDevice* config) : config(config) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/"
      "differentiable/common/gradient-clear.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto GradientClearPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("ParamGradient")
      .isBuffer().withSize(std::max(config->gradient_buffer_size, size_t(64)))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("ParamGradientAuxiliary")
      .isBuffer().withSize(std::max(config->gradient_buffer_size *
        config->optimizer.auxiliary_multiplier(), size_t(64)))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("LossSummation")
      .isBuffer().withSize(size_t(64))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                  (uint32_t)RHI::BufferUsage::COPY_SRC |
                  (uint32_t)RHI::BufferUsage::MAP_READ)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto GradientClearPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void {
  GFX::Buffer* buffer = renderData.getBuffer("ParamGradient");
  GFX::Buffer* error_sum = renderData.getBuffer("LossSummation");
  updateBindings(context, {
    {"ParamGradients", RHI::BindingResource{{buffer->buffer.get(), 0, buffer->buffer->size()}}},
    {"u_error_sum", RHI::BindingResource{{error_sum->buffer.get(), 0, error_sum->buffer->size()}}},
  });
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&config->gradient_buffer_elements, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(uint32_t));
  if (config->gradient_buffer_size != 0) 
      encoder->dispatchWorkgroupsIndirect(config->clean_resources_device.get(), 0);
  else encoder->dispatchWorkgroups(1);
  encoder->end();
}

GradientDescentTexPass::GradientDescentTexPass(
    DifferentiableDevice* config, GFX::Texture* texture, uint32_t rid)
    : config(config), texture(texture), rid(rid) {
  char const* enum_values[] = {"0", "1", "2", "3"};
  int enum_index = 0;
  if (texture->texture->format() == RHI::TextureFormat::RGBA32_FLOAT) {
    enum_index = 3;
  }
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/"
      "differentiable/common/gradient-descent-texture.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      }, {{"DATA_TYPE_ENUM", enum_values[enum_index]}});
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto GradientDescentTexPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("ParamGradient")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto GradientDescentTexPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void {
  if (config->gradient_buffer_size == 0) return;

  context->cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor {
    (uint32_t) RHI::PipelineStages::ALL_GRAPHICS_BIT,
    (uint32_t) RHI::PipelineStages::COMPUTE_SHADER_BIT,
    0, {}, {}, {RHI::TextureMemoryBarrierDescriptor{
      texture->texture.get(), RHI::ImageSubresourceRange{
        uint32_t(RHI::TextureAspect::COLOR_BIT), 0, 1, 0, 1 },
      (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
      (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT|(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
      RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
      RHI::TextureLayout::GENERAL
    }}});
  
  GFX::Buffer* buffer = renderData.getBuffer("ParamGradient");
  updateBindings(context, {
    {"ParamGradients", RHI::BindingResource{{buffer->buffer.get(), 0, buffer->buffer->size()}}},
    {"DiffResourcesDescs", RHI::BindingResource{{config->diff_resources_device.get(), 0, config->diff_resources_device->size()}}},
    {"u_resource", RHI::BindingResource{{texture->getUAV(0,0,1)}}},
  });
  struct PushConstant {
    int width;
    int height;
    int resource_id;
    float learning_rate;
  } pconst = {
    texture->texture->width(),
    texture->texture->height(),
    rid, config->learning_rate
  };
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pconst, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(pconst));
  encoder->dispatchWorkgroups((pconst.width + 15) / 16, (pconst.height + 15) / 16, 1);
  encoder->end();
  
  context->cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor {
    (uint32_t) RHI::PipelineStages::COMPUTE_SHADER_BIT,
    (uint32_t) RHI::PipelineStages::ALL_GRAPHICS_BIT,
    0, {}, {}, {RHI::TextureMemoryBarrierDescriptor{
      texture->texture.get(), RHI::ImageSubresourceRange{
        uint32_t(RHI::TextureAspect::COLOR_BIT), 0, 1, 0, 1 },
      (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
      (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
      RHI::TextureLayout::GENERAL,
      RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL,
    }}});
}

GradientDescentPrimPass::GradientDescentPrimPass(
    DifferentiableDevice* config)
    : config(config), buffer(config->primal_resources_device.get()) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/"
      "differentiable/common/gradient-descent-primal.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),});
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto GradientDescentPrimPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("ParamGradient")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("ParamGradientAuxiliary")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto GradientDescentPrimPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void {
  if (config->primal_buffer_size == 0) return;

  context->cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor {
    (uint32_t) RHI::PipelineStages::ALL_GRAPHICS_BIT,
    (uint32_t) RHI::PipelineStages::COMPUTE_SHADER_BIT,
    0, {}, {RHI::BufferMemoryBarrierDescriptor {
      buffer, (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
      (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT|(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
    }}, {}});
    
  GFX::Buffer* gradient = renderData.getBuffer("ParamGradient");
  GFX::Buffer* auxiliary = renderData.getBuffer("ParamGradientAuxiliary");
  updateBindings(context, {
    {"u_parameter_buffer", RHI::BindingResource{{buffer, 0, buffer->size()}}},
    {"u_gradient_buffer", RHI::BindingResource{{gradient->buffer.get(), 0, gradient->buffer->size()}}},
    {"u_auxiliary_buffer", RHI::BindingResource{{auxiliary->buffer.get(), 0, auxiliary->buffer->size()}}},
  });

  struct PushConstant {
    OptimizerInstance::ParamPack optimizer;
    int param_count;
    int grad_offset;
    int iteration;
  } pconst = { 
    config->optimizer.param,
    config->primal_buffer_size,
    0,
    config->training.iteration,
  };

  const uint32_t prim_count = config->primal_buffer_size;
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pconst, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(pconst));
  encoder->dispatchWorkgroups((prim_count + 255) / 256, 1, 1);
  encoder->end();
  
  context->cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor {
    (uint32_t) RHI::PipelineStages::COMPUTE_SHADER_BIT,
    (uint32_t) RHI::PipelineStages::ALL_GRAPHICS_BIT,
    0, {}, {RHI::BufferMemoryBarrierDescriptor {
      buffer, (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
      (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT|(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
    }}, {}});
}

LossReadbackTexPass::LossReadbackTexPass(DifferentiableDevice* config)
    : config(config) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/"
      "differentiable/common/loss-accumulation.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto LossReadbackTexPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("LossSummation")
    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                 (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("LossTexture")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
      RDG::TextureInfo::ConsumeType::StorageBinding}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector; }

auto LossReadbackTexPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* loss_tex = renderData.getTexture("LossTexture");
  GFX::Buffer* error_sum = renderData.getBuffer("LossSummation");
  updateBindings(context, {
    {"u_error", RHI::BindingResource{{loss_tex->getUAV(0, 0, 1)}}},
    {"u_error_sum", RHI::BindingResource{{error_sum->buffer.get(), 0, error_sum->buffer->size()}}},
  });
  struct PushConstant {
    Math::ivec2 size;
    float gradient_multiplier;
  } pConst = {
    {int(loss_tex->texture->width()), int(loss_tex->texture->height())},
    config->training.grad_multiplier
  };
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants((void*)&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(pConst));
  encoder->dispatchWorkgroups((pConst.size.x + 15) / 16, (pConst.size.y + 15) / 16);
  encoder->end();
}

auto LossReadbackTexPass::renderUI() noexcept -> void {
  if (config->training.on_training) loss_history.push_back(loss);
  else loss_history.clear();
  if (loss_history.size() > 0)
    ImGui::PlotLines("Loss", loss_history.data(), loss_history.size(), 0, NULL,
                     0.f, FLT_MAX, ImVec2(0, 200.0f));
}

auto LossReadbackTexPass::readback(
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* error_sum = renderData.getBuffer("LossSummation");
  RHI::Device* device = RHI::RHILayer::get()->getDevice();
  std::unique_ptr<RHI::CommandEncoder> encoder =
      device->createCommandEncoder({});
  encoder->pipelineBarrier(RHI::BarrierDescriptor{
    (uint32_t) RHI::PipelineStages::COMPUTE_SHADER_BIT,
    (uint32_t) RHI::PipelineStages::HOST_BIT,
    0, {}, {RHI::BufferMemoryBarrierDescriptor{
      error_sum->buffer.get(),
      (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT,
      (uint32_t)RHI::AccessFlagBits::HOST_READ_BIT,
      0, 64}}, {}});
  device->getGraphicsQueue()->submit({encoder->finish({})});
  device->waitIdle();

  std::future<bool> mapped =
      error_sum->buffer->mapAsync((uint32_t)RHI::MapMode::READ, 0, 64);
  if (mapped.get()) {
    void* mapdata = error_sum->buffer->getMappedRange(0, 64);
    memcpy(&loss, mapdata, sizeof(float));
    error_sum->buffer->unmap();
  }
}

ForwardReferencePass::ForwardReferencePass() { RDG::Pass::init(); }

auto ForwardReferencePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
    reflector.addOutput("Primal")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
        RDG::TextureInfo::ConsumeType::StorageBinding}
        .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

MatrixSanityCheck::MatrixSanityCheck(DifferentiableDevice* diffd)
    : diffd(diffd) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/tinynn/tiny-mlp.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      }, {}, true);
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto MatrixSanityCheck::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("ParamGradient")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("output_buffer")
      .isBuffer().withSize(32 * 16 * sizeof(float))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Output")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .setLayout(RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("DBG")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .setLayout(RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Error")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .setLayout(RHI::TextureLayout::COLOR_ATTACHMENT_OPTIMAL)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto MatrixSanityCheck::execute(RDG::RenderContext* context,
                               RDG::RenderData const& renderData) noexcept
    -> void {
  RHI::Buffer* param_buffer = diffd->primal_resources_device.get();
  GFX::Buffer* param_diff_buffer = renderData.getBuffer("ParamGradient");
  GFX::Buffer* out_mat = renderData.getBuffer("output_buffer");
  GFX::Texture* output = renderData.getTexture("Output");
  GFX::Texture* debug = renderData.getTexture("DBG");
  GFX::Texture* error = renderData.getTexture("Error");
  
  RHI::Sampler* sampler = GFX::GFXManager::get()->samplerTable.fetch(
      RHI::AddressMode::CLAMP_TO_EDGE, RHI::FilterMode::NEAREST,
      RHI::MipmapFilterMode::NEAREST);

  updateBindings( context, {
    {"output_buffer", RHI::BindingResource{{out_mat->buffer.get(), 0, out_mat->buffer->size()}}},
    {"debug_texture", RHI::BindingResource{{debug->getUAV(0, 0, 1)}}},
    {"output_texture", RHI::BindingResource{{output->getUAV(0, 0, 1)}}},
    {"error_texture", RHI::BindingResource{{error->getUAV(0, 0, 1)}}},
    {"target_texture", RHI::BindingResource{{diffd->ground_truth->getSRV(0, 1, 0, 1)}, sampler}},
    {"_primal_buffer", RHI::BindingResource{{param_buffer, 0, param_buffer->size()}}},
    {"_gradient_buffer", RHI::BindingResource{{param_diff_buffer->buffer.get(), 0, param_diff_buffer->buffer->size()}}},
  });

  struct PushConstant {
    Math::ivec2 size;
    float gradient_multiplier;
  } pConst = {
    {int(output->texture->width()), int(output->texture->height())},
    diffd->training.grad_multiplier
  };

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(pConst));
  encoder->dispatchWorkgroups(4 * 4, 32 * 4, 1);
  encoder->end();
}

auto MatrixSanityCheck::on_register() noexcept -> void {

}

auto MatrixSanityCheck::get_buffer_param_count() noexcept -> uint32_t {
  return mlp.get_buffer_param_count() + feature_grid.get_buffer_param_count(); }
auto MatrixSanityCheck::get_primal_buffer_size() noexcept -> uint32_t {
  return mlp.get_buffer_param_count() + feature_grid.get_buffer_param_count(); }
auto MatrixSanityCheck::initialize_primal(
    std::span<float> primal, ParamInitializer* initializer) noexcept -> void {
  /*cnpy::npz_t param_npz = cnpy::npz_load("P:/GitProjects/slang-python/examples/inline-mlp-example/params.npz");
  cnpy::NpyArray arr_w1 = param_npz["w1"];
  cnpy::NpyArray arr_b1 = param_npz["b1"];
  cnpy::NpyArray arr_w2 = param_npz["w2"];
  cnpy::NpyArray arr_b2 = param_npz["b2"];
  cnpy::NpyArray arr_w3 = param_npz["w3"];
  cnpy::NpyArray arr_b3 = param_npz["b3"];
  cnpy::NpyArray feature_grid = param_npz["feature_grid"];
  for (int i = 0; i < 16 * 16; ++i) {
    primal[i] = arr_w1.data<float>()[i] * 0.05f;
    primal[(16 * 17) * 1 + i] = arr_w2.data<float>()[i] * 0.05f;
    primal[(16 * 17) * 2 + i] = arr_w3.data<float>()[i] * 0.05f;
  }
  for (int i = 0; i < 16; ++i) {
    primal[i * 16 + i] += 1;
    primal[(16 * 17) * 1 + i * 16 + i] += 1;
    primal[(16 * 17) * 2 + i * 16 + i] += 1;
  }
  for (int i = 0; i < 16; ++i) {
    primal[16 * 16 + i] = arr_b1.data<float>()[i];
    primal[(16 * 17) * 1 + 16 * 16 + i] = arr_b2.data<float>()[i];
    primal[(16 * 17) * 2 + 16 * 16 + i] = arr_b3.data<float>()[i];
  }
  uint32_t offset = 3 * 16 * 17;
  for (int i = 0; i < 33*33*14; ++i) {
    primal[offset + i] = feature_grid.data<float>()[i];
  }*/

  uint32_t size_mlp = mlp.get_buffer_param_count();
  uint32_t size_grid = feature_grid.get_buffer_param_count();
  mlp.initialize_primal(primal.subspan(0, size_mlp), initializer);
  feature_grid.initialize_primal(primal.subspan(size_mlp, size_grid), initializer);
}


ParamInitializer::ParamInitializer(int seed) { e.seed(seed); }
auto ParamInitializer::uniform(std::span<float> span, float min, float max) noexcept -> void {
  std::uniform_real_distribution<float> u(min, max);
  for (auto& i : span) { i = u(e); } }
auto ParamInitializer::normal(std::span<float> span, float mean, float var) noexcept -> void {
  std::normal_distribution<float> n(mean, var);
  for (auto& i : span) { i = n(e); } }
auto ParamInitializer::constant(std::span<float> span, float var) noexcept -> void {
    for (auto& i : span) { i = var; } }

// Linear layer definition
LinearLayer::LinearLayer(int C) : C(C) {}
auto LinearLayer::get_buffer_param_count() noexcept -> uint32_t { return (C * C + C); }
auto LinearLayer::initialize_primal(std::span<float> primal,
    ParamInitializer* initializer) noexcept
    -> void {
    initializer->normal(primal.subspan(0, C * C), 0.f, 0.1f);
    initializer->constant(primal.subspan(C * C, C), 0.f);
}

// MLP layer definition
MLPLayer::MLPLayer(int N, int C) : N(N), C(C) {}
auto MLPLayer::get_buffer_param_count() noexcept -> uint32_t { return N * (C * C + C); }
auto MLPLayer::initialize_primal(std::span<float> primal,
  ParamInitializer* initializer) noexcept -> void {
  LinearLayer linear(C);
  for (int i = 0; i < N; ++i) {
    auto linear_span = primal.subspan(i * (C * C + C), (C * C + C));
    //initializer->normal(linear_span.subspan(0, C * C), 0.f, 0.05);
    initializer->constant(linear_span.subspan(0, C * C), 0.f);
    initializer->constant(linear_span.subspan(C * C, C), 0.f);
    for (int j = 0; j < C; ++j) linear_span[j + j * C] += 1;
  }
}

// FeatureGrid2D layer definition
FeatureGrid2D::FeatureGrid2D(Math::ivec2 grid, int dim) : grid(grid), dim(dim) {}
auto FeatureGrid2D::get_buffer_param_count() noexcept -> uint32_t {
  return (grid.x + 1) * (grid.y + 1) * dim;
}
auto FeatureGrid2D::initialize_primal(std::span<float> primal,
    ParamInitializer* initializer) noexcept -> void {
    initializer->normal(primal, 0, 1);
}

}  // namespace SIByL::Addon::Differentiable