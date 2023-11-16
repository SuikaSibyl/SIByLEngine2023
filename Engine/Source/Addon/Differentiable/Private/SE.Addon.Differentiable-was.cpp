#include "../Public/SE.Addon.Differentiable-was.hpp"

namespace SIByL::Addon::Differentiable {
WasSimple::WasSimple() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/differentiable/"
      "was/was-simple.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  // GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  GFX::SBTsDescriptor sbt;
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};
  // sbt.callableSBT.callableRecords.insert(
  //   sbt.callableSBT.callableRecords.end(),
  //   RTCommon::get()->diffCallables.begin(),
  //   RTCommon::get()->diffCallables.end()
  //);
  RayTracingPass::init(sbt, 1);
}

auto WasSimple::reflect() noexcept -> RDG::PassReflection {
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

auto WasSimple::execute(RDG::RenderContext* context,
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

auto WasSimple::renderUI() noexcept -> void {
  // ImGui::DragInt("Max Depth", &max_depth);
  // ImGui::DragInt("SPP", &spp);
}
}  // namespace SIByL::Addon::Differentiable