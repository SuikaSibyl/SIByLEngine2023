#include "../Public/SE.Addon.Differentiable.hpp"
#include <SE.Editor.GFX.hpp>

namespace SIByL::Addon::Differentiable {

TestGTPass::TestGTPass() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "differentiable/test-gt.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto TestGTPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("Output")
      .isTexture().withSize(Math::ivec3(1280, 720, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto TestGTPass::execute(RDG::RenderContext* context,
                         RDG::RenderData const& renderData) noexcept -> void {
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  GFX::Texture* output = renderData.getTexture("Output");
  updateBindings(context, {
          {"u_output", RHI::BindingResource{{output->getUAV(0, 0, 1)}}},
      });

  int rand_seed = renderData.getUInt("FrameIdx");
  RHI::RayTracingPassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&rand_seed, (uint32_t)RHI::ShaderStages::RAYGEN, 0, sizeof(uint32_t));
  encoder->traceRays(1280, 720, 1);
  encoder->end();
}

TestADPass::TestADPass() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "differentiable/test-ad.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);

  // Load gt image
  Core::GUID gt_guid = GFX::GFXManager::get()->registerTextureResource(
      "D:/Art/Scenes/differentiable_test/gt.exr");
  gt = Core::ResourceManager::get()->getResource<GFX::Texture>(gt_guid);
}

auto TestADPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("RGBA32").isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Output")
      .isTexture().withSize(Math::vec3(1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto TestADPass::execute(RDG::RenderContext* context,
                                RDG::RenderData const& renderData) noexcept -> void {
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  RHI::Sampler* sampler = GFX::GFXManager::get()->samplerTable.fetch(
      RHI::AddressMode::CLAMP_TO_EDGE, RHI::FilterMode::NEAREST,
      RHI::MipmapFilterMode::NEAREST);

  GFX::Texture* rgba = renderData.getTexture("RGBA32");
  GFX::Texture* output = renderData.getTexture("Output");
  updateBindings(context, {
          {"u_color", RHI::BindingResource{{rgba->getUAV(0, 0, 1)}}},
          {"u_output", RHI::BindingResource{{output->getUAV(0, 0, 1)}}},
          {"u_target", RHI::BindingResource{gt->getUAV(0, 0, 1), sampler}},
  });

  struct PushConstant {
    float learning_rate;
    int learn;
    int initialize;
    int rand_seed;
  } pConst = {
      learning_rate, 
      learn, 
      initialize, 
      renderData.getUInt("FrameIdx")
  };
  initialize = false;

  RHI::RayTracingPassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0, sizeof(pConst));
  encoder->traceRays(1280, 720, 1);
  encoder->end();
}

auto TestADPass::renderUI() noexcept -> void {
  ImGui::Checkbox("learn", &learn);
  ImGui::DragFloat("learn_rate", &learning_rate);
}

AdjointRenderPass::AdjointRenderPass(DifferentiableDevice* config)
    : config(config) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/"
      "differentiable/radiative-backpropagation/adjoint-rendering.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto AdjointRenderPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("Primal")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("DeltaY")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto AdjointRenderPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void {
  if (config->ground_truth == nullptr) return;
  GFX::Texture* input = renderData.getTexture("Primal");
  GFX::Texture* delta = renderData.getTexture("DeltaY");

  RHI::Sampler* sampler = GFX::GFXManager::get()->samplerTable.fetch(
      RHI::AddressMode::CLAMP_TO_EDGE, RHI::FilterMode::NEAREST,
      RHI::MipmapFilterMode::NEAREST);

  updateBindings(context, {
    {"u_reference", RHI::BindingResource{{config->ground_truth->getSRV(0, 1, 0, 1)}, sampler}},
    {"u_input", RHI::BindingResource{input->getSRV(0, 1, 0, 1)}},
    {"u_deltaY", RHI::BindingResource{delta->getUAV(0, 0, 1)}},
  });
  struct PushConstant {
    int width;
    int height;
    int loss;
  } pConst = {
    input->texture->width(),
    input->texture->height(),
    loss_func
  };
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(PushConstant));
  encoder->dispatchWorkgroups((pConst.width + 15)/16, (pConst.height + 15)/16, 1);
  encoder->end();
}

auto AdjointRenderPass::renderUI() noexcept -> void {
  { const char* item_names[] = {"L2 Loss"};
    ImGui::Combo("Sample Mode", &loss_func, item_names,
                 IM_ARRAYSIZE(item_names), IM_ARRAYSIZE(item_names));
  }
}

RadiativeBackpropPass::RadiativeBackpropPass(DifferentiableDevice* config)
    : config(config) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/differentiable/"
      "radiative-backpropagation/radiative-backprop.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};
  sbt.callableSBT.callableRecords.insert(
    sbt.callableSBT.callableRecords.end(),
    RTCommon::get()->diffCallables.begin(),
    RTCommon::get()->diffCallables.end()
  );
  RayTracingPass::init(sbt, 1);
}

auto RadiativeBackpropPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("ParamGradient")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT
                        |(uint32_t) RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("DeltaY")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("Color")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Debug")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto RadiativeBackpropPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void {
  if (config->gradient_buffer_size == 0) return;

  GFX::Buffer* buffer = renderData.getBuffer("ParamGradient");
  GFX::Texture* deltaY = renderData.getTexture("DeltaY");
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* debug = renderData.getTexture("Debug");
  
  updateBindings(context, {
    {"ParamGradients", RHI::BindingResource{{buffer->buffer.get(), 0, buffer->buffer->size()}}},
    {"DiffResourcesDescs", RHI::BindingResource{{config->diff_resources_device.get(), 0, config->diff_resources_device->size()}}},
    {"DiffableTextureIndices", RHI::BindingResource{{config->texture_indices_device.get(), 0, config->texture_indices_device->size()}}},
    {"u_deltaY", RHI::BindingResource{{deltaY->getUAV(0, 0, 1)}}},
    {"u_color", RHI::BindingResource{{color->getUAV(0, 0, 1)}}},
    {"u_debug", RHI::BindingResource{{debug->getUAV(0, 0, 1)}}},
  });
  
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  struct PushConst {
    int rand_seed;
    int max_depth;
    int spp;
  } pConst;
  pConst.rand_seed = renderData.getUInt("FrameIdx");
  pConst.max_depth = max_depth;
  pConst.spp = spp;
  RHI::RayTracingPassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0, sizeof(pConst));
  encoder->traceRays(1280, 720, 1);
  encoder->end();
}

auto RadiativeBackpropPass::renderUI() noexcept -> void {
  ImGui::DragInt("Max Depth", &max_depth);
  ImGui::DragInt("SPP", &spp);
}

ReparamSimple::ReparamSimple() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/differentiable/"
      "reparam/reparam-simple.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  //GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  GFX::SBTsDescriptor sbt;
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};
  //sbt.callableSBT.callableRecords.insert(
  //  sbt.callableSBT.callableRecords.end(),
  //  RTCommon::get()->diffCallables.begin(),
  //  RTCommon::get()->diffCallables.end()
  //);
  RayTracingPass::init(sbt, 1);
}

auto ReparamSimple::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("Color")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("LossTexture")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::R32_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("ParamGradient")
    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT |
                        (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
            .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto ReparamSimple::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* loss = renderData.getTexture("LossTexture");
  
  updateBindings(context, {
    {"u_color", RHI::BindingResource{{color->getUAV(0, 0, 1)}}},
    {"u_loss", RHI::BindingResource{{loss->getUAV(0, 0, 1)}}},
  });

  GFX::Buffer* gradient = renderData.getBuffer("ParamGradient");
  updateBinding(context, "u_buffer",
                RHI::BindingResource{{gradient->buffer.get(), 0, gradient->buffer->size()}});

  int rand_seed = renderData.getUInt("FrameIdx");

  RHI::RayTracingPassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&rand_seed, (uint32_t)RHI::ShaderStages::RAYGEN, 0, sizeof(int));
  encoder->traceRays(1280, 720, 1);
  encoder->end();
}

auto ReparamSimple::renderUI() noexcept -> void {
  //ImGui::DragInt("Max Depth", &max_depth);
  //ImGui::DragInt("SPP", &spp);
}

ReparamInit::ReparamInit() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/"
      "differentiable/reparam/reparam-begin.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto ReparamInit::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("ParamGradient")
      .isBuffer().withSize(32)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto ReparamInit::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void {

  GFX::Buffer* gradient = renderData.getBuffer("ParamGradient");
  updateBinding(context, "u_buffer",
                RHI::BindingResource{{gradient->buffer.get(), 0, gradient->buffer->size()}});
  struct PushConstant {
    float learning_rate;
    int initialize;
    float init_value_x;
    float init_value_y;
    float init_value_z;
  } pConst = {learning_rate, init ? 1 : 0, initial_value.x, initial_value.y,
              initial_value.z};

  if (init) {
    init = false;
  }
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(PushConstant));
  encoder->dispatchWorkgroups(1, 1, 1);
  encoder->end();
}

auto ReparamInit::renderUI() noexcept -> void {
  ImGui::DragFloat("Initial Value-x", &initial_value.x);
  ImGui::DragFloat("Initial Value-y", &initial_value.y);
  ImGui::DragFloat("Initial Value-z", &initial_value.z);
  ImGui::DragFloat("Learning Rate", &learning_rate, 0, -1, 1, "%.8f");
  ImGui::Checkbox("Init", &init);
}

AutoDiffGraph::AutoDiffGraph(DifferentiableDevice* configure) {
  
  addPass(std::make_unique<Addon::Differentiable::GradientClearPass>(configure), "GradientClear Pass");

  addPass(std::make_unique<Addon::Differentiable::MatrixSanityCheck>(configure), "MatrixSanity Pass");
  addEdge("GradientClear Pass", "ParamGradient", "MatrixSanity Pass", "ParamGradient");
  addModule(reinterpret_cast<Addon::Differentiable::MatrixSanityCheck*>(getPass("MatrixSanity Pass")));

  ////// Radiative backgropagation pass
  ////addPass(std::make_unique<Addon::Differentiable::ForwardReferencePass>(), "ForwardRef Pass");
  ////addPass(std::make_unique<Addon::Differentiable::AdjointRenderPass>(configure), "AdjointRender Pass");
  ////addEdge("ForwardRef Pass", "Primal", "AdjointRender Pass", "Primal");
  ////addPass(std::make_unique<Addon::Differentiable::RadiativeBackpropPass>(configure), "RadiativeBackprop Pass");
  ////addEdge("GradientClear Pass", "ParamGradient", "RadiativeBackprop Pass", "ParamGradient");
  ////addEdge("AdjointRender Pass", "DeltaY", "RadiativeBackprop Pass", "DeltaY");
  ////addEdge("ForwardRef Pass", "Primal", "RadiativeBackprop Pass", "Color");
  ////
  ////for (int i = 0; i < configure->diff_resources_host.size(); ++i) {
  ////  DiffResourceDesc const& desc = configure->diff_resources_host[i];
  ////  if ((desc.data_flag & 0b11) == (uint32_t)DiffResourceType::DIFF_RESOURCE_TYPE_TEXTURE) {
  ////    GFX::Texture* texture = configure->diff_textures_host[i];
  ////    std::string pass_name = "GradientDescent Tex-" + std::to_string(i) + " Pass";
  ////    addPass(std::make_unique<Addon::Differentiable::GradientDescentTexPass>(configure, texture, i), pass_name);
  ////    addEdge("RadiativeBackprop Pass", "ParamGradient", pass_name, "ParamGradient");
  ////  }
  ////}

  //addPass(std::make_unique<Addon::Differentiable::ReparamInit>(), "Init Pass");
  ////addPass(std::make_unique<Addon::Differentiable::ReparamSimple>(), "Reparam Pass");
  ////addEdge("Init Pass", "ParamGradient", "Reparam Pass", "ParamGradient");
  //addPass(std::make_unique<Addon::Differentiable::WasSimple>(), "Reparam Pass");
  //addEdge("Init Pass", "ParamGradient", "Reparam Pass", "ParamGradient");

  addPass(std::make_unique<Addon::Differentiable::GradientDescentPrimPass>(configure), "GradientDescent-Prim Pass");
  addEdge("GradientClear Pass", "ParamGradient", "GradientDescent-Prim Pass", "ParamGradient");
  addEdge("GradientClear Pass", "ParamGradientAuxiliary", "GradientDescent-Prim Pass", "ParamGradientAuxiliary");
  // Loss readback to get the total loss summation
  addPass(std::make_unique<Addon::Differentiable::LossReadbackTexPass>(configure), "LossReadback Pass");
  addEdge("GradientClear Pass", "LossSummation", "LossReadback Pass", "LossSummation");
  addEdge("MatrixSanity Pass", "Error", "LossReadback Pass", "LossTexture");

  ////addEdge("GradientClear Pass", "ParamGradient", "Reparam Pass", "ParamGradient");
  //////setting.buffer_size = 1024;
  //////addPass(std::make_unique<Addon::Differentiable::GradientClearPass>(&setting), "GradientClear Pass");
  //////addPass(std::make_unique<Addon::Differentiable::RadiativeBackprop>(&setting), "RadiativeBackprop Pass");
  //////addEdge("GradientClear Pass", "ParamGradient", "RadiativeBackprop Pass", "ParamGradient");
  ////
  //////addPass(std::make_unique<Addon::Differentiable::TestADPass>(), "TestAD Pass");
  //markOutput("Reparam Pass", "Color");
  ////markOutput("RadiativeBackprop Pass", "Color");
  markOutput("MatrixSanity Pass", "Output");
}

auto FusedADPipeline::execute(RHI::CommandEncoder* encoder) noexcept -> void {
  if (diffdevice.training.on_training) {
    pGraph->execute(encoder);
    diffdevice.optimizer.param.step_count = diffdevice.training.iteration;
    diffdevice.training.iteration++; } }
auto FusedADPipeline::getActiveGraphs() noexcept -> std::vector<RDG::Graph*> {
  return {pGraph}; }
auto FusedADPipeline::getOutput() noexcept -> GFX::Texture* {
  return pGraph->getOutput(); }

AutoDiffPipeline::AutoDiffPipeline() {
  graph = AutoDiffGraph(&diffdevice);
  pGraph = &graph; }
auto AutoDiffPipeline::renderUI() noexcept -> void {
  ImGui::Text("Training"); ImGui::SameLine();
  if (diffdevice.training.on_training) {
    if (ImGui::Button("stop")) diffdevice.training.on_training = false;
  } else {
    if (ImGui::Button("begin")) {
      diffdevice.training.on_training = true;
      diffdevice.training.start_training = true;
      diffdevice.training.iteration = 0;
    }
  }
  ImGui::Separator();

  ImGui::Text("Optimizer");
  if (diffdevice.training.on_training) ImGui::BeginDisabled();
  diffdevice.optimizer.renderUI();
  if (diffdevice.training.on_training) ImGui::EndDisabled();
  ImGui::Separator();

  Editor::drawCustomColume("Reference", 100, [&]() {
    if (ImGui::Button("Load")) {
      std::string path = Editor::ImGuiLayer::get()->rhiLayer->getRHILayerDescriptor().windowBinded->openFile("");
      if (path != "") {
        Core::GUID guid = GFX::GFXManager::get()->registerTextureResource(path.c_str());
        diffdevice.ground_truth =
            Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
        if (diffdevice.ground_truth != nullptr) {
          diffdevice.gt_filepath = path;
        }
      }
    }
    if (diffdevice.ground_truth) {
      // show ground truth image
      GFX::Texture* tex = diffdevice.ground_truth;
      float const texw = (float)tex->texture->width();
      float const texh = (float)tex->texture->height();
      float const wa = std::max(1.f, ImGui::GetContentRegionAvail().x - 15) / texw;
      float const ha = 1; float a = std::min(1.f, std::min(wa, ha));
      ImGui::Image(Editor::TextureUtils::getImGuiTexture(tex->guid)->getTextureID(),
          {a * texw, a * texh}, {0, 0}, {1, 1});
    }
  });
  Editor::drawCustomColume("Configure", 100, [&]() {
    ImGui::DragFloat("Learning Rate", &diffdevice.learning_rate, 0.01f, -1, 1, "%.5f");
  });
  Editor::drawCustomColume("Rebuild", 100, [&]() {
    // rebuild the whole pipeline!
    if (ImGui::Button("build")) {
      RHI::RHILayer::get()->getDevice()->waitIdle();
      diffdevice.primal_buffer_size = 0;
      // gather all differentiable resources.
      SRenderer* srenderer = SRenderer::singleton;
      auto& textures = srenderer->sceneDataPack.texture_record;
      diffdevice.texture_indices_host.resize(
          srenderer->sceneDataPack.unbinded_textures.size());
      for (auto& index : diffdevice.texture_indices_host) index = -1;
      int offset = 0;
      int data_offset = 0;
      for (auto pair : textures) {
        GFX::Texture* texture = pair.first;
        if (texture->differentiable_channels != 0) {
          diffdevice.texture_indices_host[pair.second] = offset++;
          const std::bitset<4> channels(texture->differentiable_channels);
          const uint32_t available_channel_cnt = channels.count();
          const int mipmap_level = 1 + int(floor(log2(texture->texture->width())));
          const int sum_elements = ((1 << (mipmap_level * 2)) - 1) / 3;
          const uint32_t resource_size = sum_elements * sizeof(float) * available_channel_cnt;
          diffdevice.diff_resources_host.emplace_back(
              resource_size, data_offset, 
              texture->texture->width() | (texture->texture->height() << 16),
              texture->differentiable_channels << 2 | 1
          );
          diffdevice.diff_textures_host.push_back(texture);
          data_offset += diffdevice.diff_resources_host.back().data_size;
          data_offset = (data_offset * 64 + 63) / 64;   // alignment
        }
      }

      // gather all NN weights
      {
        ParamInitializer initializer = ParamInitializer(0);
        for (auto& mode : graph.modules) {
          // buffer resource registration
          const uint32_t gradient_size = mode->get_buffer_param_count() * sizeof(float);
          diffdevice.diff_resources_host.emplace_back(gradient_size, data_offset, 0, 0);
          mode->gradient_offset = data_offset;
          data_offset += gradient_size;
          const uint32_t primal_size = mode->get_primal_buffer_size();
          const uint32_t primal_offset = diffdevice.primal_buffer_size;
          if (primal_size > 0) {
            diffdevice.primal_buffer_size += primal_size;
            diffdevice.primal_buffer_host.resize(primal_offset + primal_size);
            mode->primal_offset = primal_offset;
            mode->initialize_primal(
                std::span<float>(diffdevice.primal_buffer_host)
                    .subspan(primal_offset, primal_size), &initializer);
          }
        }
      }

      // setting global configuration data
      diffdevice.gradient_buffer_size = data_offset;
      diffdevice.gradient_buffer_elements = data_offset / sizeof(float);
      // create device buffer resources
      RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
      if (diffdevice.gradient_buffer_size == 0) {
        //Core::LogManager::Error("No differentiable resources!");
      } else {
        diffdevice.texture_indices_device = device->createDeviceLocalBuffer(
          diffdevice.texture_indices_host.data(), diffdevice.texture_indices_host.size() * sizeof(int),
          (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS | (uint32_t)RHI::BufferUsage::STORAGE);
        diffdevice.diff_resources_device = device->createDeviceLocalBuffer(
          diffdevice.diff_resources_host.data(), diffdevice.diff_resources_host.size() * sizeof(DiffResourceDesc),
          (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS | (uint32_t)RHI::BufferUsage::STORAGE);      
        std::vector<int> indices = { int(diffdevice.gradient_buffer_elements + 255) / 256, 1, 1, 0};
        diffdevice.clean_resources_device = device->createDeviceLocalBuffer(
            indices.data(), indices.size() * sizeof(int),
            (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
            (uint32_t)RHI::BufferUsage::STORAGE |
            (uint32_t)RHI::BufferUsage::INDIRECT);
      }
      if (diffdevice.primal_buffer_host.size() > 0) {
        diffdevice.primal_resources_device = device->createDeviceLocalBuffer(
            diffdevice.primal_buffer_host.data(),
            diffdevice.primal_buffer_host.size() * sizeof(float),
            (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
            (uint32_t)RHI::BufferUsage::STORAGE |
            (uint32_t)RHI::BufferUsage::INDIRECT);      
      }
      // recreate the graph and the pipeline
      graph = AutoDiffGraph(&diffdevice);
      build();  // rebuild the pipeline
    }
  });
}
}