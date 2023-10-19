#include "../Public/SE.Addon.Differentiable.hpp"

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
}