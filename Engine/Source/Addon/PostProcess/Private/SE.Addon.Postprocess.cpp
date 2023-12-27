#include "../Public/SE.Addon.Postprocess.hpp"

namespace SIByL::Addon::Postprocess {
AccumulatePass::AccumulatePass() {
  Core::GUID comp = GFX::GFXManager::get()->registerShaderModuleResource(
      "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/"
      "accumulate_comp.spv",
      {nullptr, RHI::ShaderStages::COMPUTE});
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto AccumulatePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("Output")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .setSubresource(0, 1, 0, 1)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("Input")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .setSubresource(0, 1, 0, 1)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInternal("LastSum")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .setSubresource(0, 1, 0, 1)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto AccumulatePass::execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* output = renderData.getTexture("Output");
  GFX::Texture* input = renderData.getTexture("Input");
  GFX::Texture* sum = renderData.getTexture("LastSum");

  getBindGroup(context, 0)
      ->updateBinding(std::vector<RHI::BindGroupEntry>{
          RHI::BindGroupEntry{0, RHI::BindingResource{input->getUAV(0, 0, 1)}},
          RHI::BindGroupEntry{1, RHI::BindingResource{sum->getUAV(0, 0, 1)}},
          RHI::BindGroupEntry{2, RHI::BindingResource{output->getUAV(0, 0, 1)}},
      });

  RHI::ComputePassEncoder* encoder = beginPass(context);

  uint32_t width = input->texture->width();
  uint32_t height = input->texture->height();
  uint32_t batchIdx = renderData.getUInt("AccumIdx");

  prepareDispatch(context);

  pConst.resolution = Math::uvec2{width, height};
  if (batchIdx == 0) {
    pConst.gAccumCount = 0;
  }
  pConst.gMovingAverageMode = pConst.gAccumCount > 0 ? 1 : 0;

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((width + 15) / 16, (height + 15) / 16, 1);

  if (maxAccumCount == 0 || pConst.gAccumCount < maxAccumCount) {
    pConst.gAccumCount++;
  }

  encoder->end();
}

auto AccumulatePass::renderUI() noexcept -> void {
  ImGui::DragInt("Max Accum", &maxAccumCount, 1, 0);
  if (ImGui::Button("Reset")) {
    pConst.gAccumCount = 0;
  }
  ImGui::SameLine();
  bool useAccum = pConst.gAccumulate != 0;
  if (ImGui::Checkbox("Use Accum", &useAccum)) {
    pConst.gAccumulate = useAccum ? 1 : 0;
  }
  // pConst.gAccumulate = useAccum ? 1 : 0;
  ImGui::Text(
      std::string("Accumulated Count: " + std::to_string(pConst.gAccumCount))
          .c_str());
}

ToneMapperPass::ToneMapperPass() {
  auto [frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/postprocess/"
      "tone-mapper.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RDG::FullScreenPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto ToneMapperPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("Input")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT)
                   .setSubresource(0, 1, 0, 1));
  reflector.addOutput("Output")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA8_UNORM)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(0));
  return reflector;
}

auto ToneMapperPass::execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* input = renderData.getTexture("Input");
  GFX::Texture* output = renderData.getTexture("Output");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{output->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{},
  };

  updateBindings(context, {
      {"s_source", RHI::BindingResource{input->getUAV(0, 0, 1)}},
  });

  struct PushConstant {
    Math::ivec2 resolution;
    float exposure;
  } pConst;
  pConst.resolution = Math::ivec2{1280, 720};
  pConst.exposure = exposure;
  RHI::RenderPassEncoder* encoder = beginPass(context, output);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::FRAGMENT, 0,
                         sizeof(PushConstant));
  dispatchFullScreen(context);
  encoder->end();
}

auto ToneMapperPass::renderUI() noexcept -> void {
  ImGui::DragFloat("Manual Exposure", &exposure, 0.01f);
}

BlendPass::BlendPass() {
  auto [frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/postprocess/"
      "blend-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RDG::FullScreenPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto BlendPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("Base").isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{RDG::TextureInfo::ConsumeType::StorageBinding}
        .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT)
        .setSubresource(0, 1, 0, 1));
  reflector.addInput("Blend").isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{RDG::TextureInfo::ConsumeType::StorageBinding}
        .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT)
        .setSubresource(0, 1, 0, 1));
  reflector.addOutput("Output").isTexture()
      .withSize(Math::vec3(1, 1, 1)).withFormat(RHI::TextureFormat::RGBA8_UNORM)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{RDG::TextureInfo::ConsumeType::ColorAttachment}
        .setAttachmentLoc(0));
  return reflector;
}

auto BlendPass::execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* input = renderData.getTexture("Input");
  GFX::Texture* blend = renderData.getTexture("Blend");
  GFX::Texture* output = renderData.getTexture("Output");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{output->getRTV(0, 0, 1),
        nullptr, {0, 0, 0, 1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{},
  };

  updateBindings(context, {
      {"u_input", RHI::BindingResource{input->getUAV(0, 0, 1)}},
      {"u_blend", RHI::BindingResource{blend->getUAV(0, 0, 1)}},
  });

  struct PushConstant {
    Math::ivec2 resolution;
    uint32_t mode;
    uint32_t rand_seed;
    float alpha;
  } pConst;
  pConst.resolution = {int(output->texture->width()),
                       int(output->texture->height())};
  pConst.mode = mode; pConst.alpha = alpha;
  pConst.rand_seed = renderData.getUInt("FrameIdx");
  RHI::RenderPassEncoder* encoder = beginPass(context, output);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::FRAGMENT, 0, sizeof(PushConstant));
  dispatchFullScreen(context);
  encoder->end();
}

auto BlendPass::renderUI() noexcept -> void {
  int combo_mode = int(mode);
  const char* kind_names[] = {"Normal", "Dissolve", "Darken", "Multiply"};
  ImGui::Combo("Type", &combo_mode, kind_names, 
      IM_ARRAYSIZE(kind_names), IM_ARRAYSIZE(kind_names));
  mode = uint32_t(combo_mode);
  ImGui::DragFloat("Alpha", &alpha, 0.01f, 0.f, 1.f);
}
}