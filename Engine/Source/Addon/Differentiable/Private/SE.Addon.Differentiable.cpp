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

  GFX::Sampler* sampler = Core::ResourceManager::get()->getResource<GFX::Sampler>(
          GFX::GFXManager::get()->commonSampler.clamp_nearest);
  GFX::Texture* rgba = renderData.getTexture("RGBA32");
  GFX::Texture* output = renderData.getTexture("Output");
  updateBindings(context, {
          {"u_color", RHI::BindingResource{{rgba->getUAV(0, 0, 1)}}},
          {"u_output", RHI::BindingResource{{output->getUAV(0, 0, 1)}}},
          {"u_target", RHI::BindingResource{gt->getUAV(0, 0, 1), sampler->sampler.get()}},
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

GradientClearPass::GradientClearPass(DifferentiableConfigure* config) : config(config) {
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
      .isBuffer().withSize(config->gradient_buffer_size)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto GradientClearPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void {
  if (config->gradient_buffer_size == 0) return;
  GFX::Buffer* buffer = renderData.getBuffer("ParamGradient");
  updateBinding(context, "ParamGradients",
                RHI::BindingResource{{buffer->buffer.get(), 0, buffer->buffer->size()}});
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&config->gradient_buffer_elements, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(uint32_t));
  encoder->dispatchWorkgroupsIndirect(config->clean_resources_device.get(), 0);
  encoder->end();
}

GradientDescentTexPass::GradientDescentTexPass(
    DifferentiableConfigure* config, GFX::Texture* texture, uint32_t rid)
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

AdjointRenderPass::AdjointRenderPass(DifferentiableConfigure* config)
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
  updateBindings(context, {
    {"u_reference", RHI::BindingResource{{config->ground_truth->getSRV(0, 1, 0, 1)},
        Core::ResourceManager::get()->getResource<GFX::Sampler>(
            GFX::GFXManager::get()->commonSampler.clamp_nearest)->sampler.get()}},
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

RadiativeBackpropPass::RadiativeBackpropPass(DifferentiableConfigure* config)
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
  } pConst;
  pConst.rand_seed = renderData.getUInt("FrameIdx");
  pConst.max_depth = max_depth;
  RHI::RayTracingPassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0, sizeof(pConst));
  encoder->traceRays(1280, 720, 1);
  encoder->end();
}

auto RadiativeBackpropPass::renderUI() noexcept -> void {
  ImGui::DragInt("Max Depth", &max_depth);
}

AutoDiffGraph::AutoDiffGraph(DifferentiableConfigure* configure) {
  addPass(std::make_unique<Addon::Differentiable::GradientClearPass>(configure), "GradientClear Pass");
  // Radiative backgropagation pass
  addPass(std::make_unique<Addon::Differentiable::ForwardReferencePass>(), "ForwardRef Pass");
  addPass(std::make_unique<Addon::Differentiable::AdjointRenderPass>(configure), "AdjointRender Pass");
  addEdge("ForwardRef Pass", "Primal", "AdjointRender Pass", "Primal");
  addPass(std::make_unique<Addon::Differentiable::RadiativeBackpropPass>(configure), "RadiativeBackprop Pass");
  addEdge("GradientClear Pass", "ParamGradient", "RadiativeBackprop Pass", "ParamGradient");
  addEdge("AdjointRender Pass", "DeltaY", "RadiativeBackprop Pass", "DeltaY");
  addEdge("ForwardRef Pass", "Primal", "RadiativeBackprop Pass", "Color");
  
  for (int i = 0; i < configure->diff_resources_host.size(); ++i) {
    DiffResourceDesc const& desc = configure->diff_resources_host[i];
    if ((desc.data_flag & 0b11) == (uint32_t)DiffResourceType::DIFF_RESOURCE_TYPE_TEXTURE) {
      GFX::Texture* texture = configure->diff_textures_host[i];
      std::string pass_name = "GradientDescent Tex-" + std::to_string(i) + " Pass";
      addPass(std::make_unique<Addon::Differentiable::GradientDescentTexPass>(configure, texture, i), pass_name);
      addEdge("RadiativeBackprop Pass", "ParamGradient", pass_name, "ParamGradient");
    }
  }

  //setting.buffer_size = 1024;
  //addPass(std::make_unique<Addon::Differentiable::GradientClearPass>(&setting), "GradientClear Pass");
  //addPass(std::make_unique<Addon::Differentiable::RadiativeBackprop>(&setting), "RadiativeBackprop Pass");
  //addEdge("GradientClear Pass", "ParamGradient", "RadiativeBackprop Pass", "ParamGradient");
  
  //addPass(std::make_unique<Addon::Differentiable::TestADPass>(), "TestAD Pass");
  markOutput("RadiativeBackprop Pass", "Color");
}

auto AutoDiffPipeline::renderUI() noexcept -> void {
  Editor::drawCustomColume("Reference", 100, [&]() {
    if (ImGui::Button("Load")) {
      std::string path = Editor::ImGuiLayer::get()->rhiLayer->getRHILayerDescriptor().windowBinded->openFile("");
      if (path != "") {
        Core::GUID guid = GFX::GFXManager::get()->registerTextureResource(path.c_str());
        configure.ground_truth = Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
        if (configure.ground_truth != nullptr) {
          configure.gt_filepath = path;
        }
      }
    }
    if (configure.ground_truth) {
      // show ground truth image
      GFX::Texture* tex = configure.ground_truth;
      float const texw = (float)tex->texture->width();
      float const texh = (float)tex->texture->height();
      float const wa = std::max(1.f, ImGui::GetContentRegionAvail().x - 15) / texw;
      float const ha = 1; float a = std::min(1.f, std::min(wa, ha));
      ImGui::Image(Editor::TextureUtils::getImGuiTexture(tex->guid)->getTextureID(),
          {a * texw, a * texh}, {0, 0}, {1, 1});
    }
  });
  Editor::drawCustomColume("Configure", 100, [&]() {
    ImGui::DragFloat("Learning Rate", &configure.learning_rate, 0.01f, -1, 1, "%.5f");
  });
  Editor::drawCustomColume("Rebuild", 100, [&]() {
    // rebuild the whole pipeline!
    if (ImGui::Button("build")) {
      RHI::RHILayer::get()->getDevice()->waitIdle();
      // gather all differentiable resources.
      SRenderer* srenderer = SRenderer::singleton;
      auto& textures = srenderer->sceneDataPack.texture_record;
      configure.texture_indices_host.resize(
          srenderer->sceneDataPack.unbinded_textures.size());
      for (auto& index : configure.texture_indices_host) index = -1;
      int offset = 0;
      int data_offset = 0;
      for (auto pair : textures) {
        GFX::Texture* texture = pair.first;
        if (texture->differentiable_channels != 0) {
          configure.texture_indices_host[pair.second] = offset++;
          const std::bitset<4> channels(texture->differentiable_channels);
          const uint32_t available_channel_cnt = channels.count();
          configure.diff_resources_host.emplace_back(
              texture->texture->width() * texture->texture->height() * sizeof(float) * available_channel_cnt,
              data_offset, texture->texture->width() | (texture->texture->height() << 16),
              texture->differentiable_channels << 2 | 1
          );
          configure.diff_textures_host.push_back(texture);
          data_offset += configure.diff_resources_host.back().data_size;
          data_offset = (data_offset * 64 + 63) / 64;   // alignment
        }
      }

      // setting global configuration data
      configure.gradient_buffer_size = data_offset;
      configure.gradient_buffer_elements = data_offset / sizeof(float);
      // create device buffer resources
      if (configure.gradient_buffer_size == 0) {
        Core::LogManager::Error("No differentiable resources!"); return;
      } else {
        RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
        configure.texture_indices_device = device->createDeviceLocalBuffer(
          configure.texture_indices_host.data(), configure.texture_indices_host.size() * sizeof(int),
          (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS | (uint32_t)RHI::BufferUsage::STORAGE);
        configure.diff_resources_device = device->createDeviceLocalBuffer(
          configure.diff_resources_host.data(), configure.diff_resources_host.size() * sizeof(DiffResourceDesc),
          (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS | (uint32_t)RHI::BufferUsage::STORAGE);      
        std::vector<int> indices = { int(data_offset / sizeof(float) + 255) / 256, 1, 1, 0};
        configure.clean_resources_device = device->createDeviceLocalBuffer(
            indices.data(), indices.size() * sizeof(int),
            (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
            (uint32_t)RHI::BufferUsage::STORAGE |
            (uint32_t)RHI::BufferUsage::INDIRECT);      
      }
      // recreate the graph and the pipeline
      graph = AutoDiffGraph(&configure);
      build();  // rebuild the pipeline
    }
  });
}
}