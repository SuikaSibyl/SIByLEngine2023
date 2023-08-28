#include "../Public/SE.Addon.ASVGF.hpp"
#include <Passes/FullScreenPasses/SE.SRenderer-Blit.hpp>
#include <SE.Addon.GBuffer.hpp>

namespace SIByL::Addon::ASVGF {
Prelude::Prelude() { RDG::DummyPass::init(); }

auto Prelude::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("GradSamplePosPrev")
      .isTexture().withSize(Math::vec3(1.f / 3))
      .withFormat(RHI::TextureFormat::R8_UINT);
  reflector.addOutput("HFPrev")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::R32_UINT);
  reflector.addOutput("SpecPrev")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::R32_UINT);
  reflector.addOutput("VBufferPrev")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::RGBA32_UINT);
  reflector.addOutput("RandPrev")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::R32_UINT);
  return reflector;
}

GradientReprojection::GradientReprojection() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/asvgf/"
      "asvgf-gradient-reproject.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto GradientReprojection::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  GBufferUtils::addGBufferInputOutput(
      reflector, (uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT);
  GBufferUtils::addPrevGbufferInputOutput(
      reflector, (uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT);

  // Writeonly
  reflector.addOutput("GradSamplePos")
      .isTexture().withSize(Math::vec3(1.f / 3))
      .withFormat(RHI::TextureFormat::R8_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("IsCorrelated")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::R8_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("RandSeed")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("HfSpecLumPrev")
      .isTexture().withSize(Math::vec3(1.f / 3))
      .withFormat(RHI::TextureFormat::RG16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("VBuffer")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  // Readonly
  reflector.addInputOutput("GradSamplePosPrev")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("HFPrev")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("SpecPrev")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("VBufferPrev")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("RandPrev")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto GradientReprojection::execute(RDG::RenderContext* context,
                                    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* gsp = renderData.getTexture("GradSamplePos");
  GFX::Texture* hf_spec_lum_prev = renderData.getTexture("HfSpecLumPrev");
  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* rand = renderData.getTexture("RandSeed");
  GFX::Texture* rand_prev = renderData.getTexture("RandPrev");
  GFX::Texture* correlate = renderData.getTexture("IsCorrelated");

  GFX::Texture* gsp_prev = renderData.getTexture("GradSamplePosPrev");
  GFX::Texture* hf_prev = renderData.getTexture("HFPrev");
  GFX::Texture* spec_prev = renderData.getTexture("SpecPrev");
  GFX::Texture* vbuffer_prev = renderData.getTexture("VBufferPrev");

  // Bind common for RT
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  
  GBufferUtils::bindGBufferResource(this, context, renderData);
  GBufferUtils::bindPrevGBufferResource(this, context, renderData);

  updateBindings(
      context,
      std::vector<std::pair<std::string, RHI::BindingResource>>{
          {"u_GradSamplePos", RHI::BindingResource{gsp->getUAV(0, 0, 1)}},
          {"u_HfSpecLum_prev", RHI::BindingResource{hf_spec_lum_prev->getUAV(0, 0, 1)}},
          {"u_VBuffer", RHI::BindingResource{vbuffer->getUAV(0, 0, 1)}},
          {"u_GradSamplePos_prev", RHI::BindingResource{gsp_prev->getUAV(0, 0, 1)}},
          {"u_HF_prev", RHI::BindingResource{hf_prev->getUAV(0, 0, 1)}},
          {"u_Spec_prev", RHI::BindingResource{spec_prev->getUAV(0, 0, 1)}},
          {"u_VBuffer_prev", RHI::BindingResource{vbuffer_prev->getUAV(0, 0, 1)}},
          {"u_RNGSeed", RHI::BindingResource{rand->getUAV(0, 0, 1)}},
          {"u_RNGSeed_prev", RHI::BindingResource{rand_prev->getUAV(0, 0, 1)}},
          {"u_IsCorrelated", RHI::BindingResource{correlate->getUAV(0, 0, 1)}},
      });

  struct PushConstant {
    uint32_t seed;
    uint32_t init_rand;
  } pConst = {0, 0};
  if (init_rand) {
    init_rand = false;
    pConst.init_rand = 1;
  }

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((1280 + 23) / 24, (720 + 23) / 24, 1);
  encoder->end();
}

GradientImagePass::GradientImagePass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/asvgf/"
      "asvgf-gradient-img.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto GradientImagePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  // Writeonly
  reflector.addOutput("GradHFSpec")
      .isTexture().withSize(Math::vec3(1.f / 3))
      .withFormat(RHI::TextureFormat::RG16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("GradHFSpecBack")
      .isTexture().withSize(Math::vec3(1.f / 3))
      .withFormat(RHI::TextureFormat::RG16_FLOAT);
  // Readonly
  reflector.addInputOutput("GradSamplePos")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("HfSpecLumPrev")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("HF")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("Spec")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto GradientImagePass::execute(RDG::RenderContext* context,
                                    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* grad = renderData.getTexture("GradHFSpec");
  GFX::Texture* gsp = renderData.getTexture("GradSamplePos");
  GFX::Texture* hf_spec_lum_prev = renderData.getTexture("HfSpecLumPrev");
  GFX::Texture* hf = renderData.getTexture("HF");
  GFX::Texture* spec = renderData.getTexture("Spec");

  updateBindings(
      context,
      std::vector<std::pair<std::string, RHI::BindingResource>>{
          {"u_grad_HF_SPEC", RHI::BindingResource{grad->getUAV(0, 0, 1)}},
          {"u_GradSamplePos", RHI::BindingResource{gsp->getUAV(0, 0, 1)}},
          {"u_HFSpecLum_prev", RHI::BindingResource{hf_spec_lum_prev->getUAV(0, 0, 1)}},
          {"u_HF", RHI::BindingResource{hf->getUAV(0, 0, 1)}},
          {"u_Spec", RHI::BindingResource{spec->getUAV(0, 0, 1)}},
      });
  
  Math::uvec2 resolution = {hf->texture->width(), hf->texture->height()};

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&resolution, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(resolution));
  encoder->dispatchWorkgroups((1280 + 15) / 16, (720 + 15) / 16, 1);
  encoder->end();
}

GradientAtrousPass::GradientAtrousPass(int iter) : iteration(iter) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/asvgf/"
      "asvgf-gradient-atrous.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto GradientAtrousPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  // Writeonly
  reflector.addInputOutput("GradHFSpecPong")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  // Readonly
  reflector.addInputOutput("GradHFSpecPing")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto GradientAtrousPass::execute(RDG::RenderContext* context,
                                    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* ping = renderData.getTexture("GradHFSpecPing");
  GFX::Texture* pong = renderData.getTexture("GradHFSpecPong");
    
  updateBindings(context,
      std::vector<std::pair<std::string, RHI::BindingResource>>{
          {"u_grad_HF_SPEC_prev", RHI::BindingResource{ping->getUAV(0, 0, 1)}},
          {"u_grad_HF_SPEC", RHI::BindingResource{pong->getUAV(0, 0, 1)}},
      });
  
  struct PushConstants {
    Math::ivec2 resolution;
    int iteration;
  } pConst = {{1280, 720}, iteration};
  Math::ivec2 grad_size = {int(std::ceil(1280.f / 3)),
                           int(std::ceil(720.f / 3))};
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstants));
  encoder->dispatchWorkgroups((grad_size.x + 15) / 16, (grad_size.y + 15) / 16,
                              1);
  encoder->end();
}

TemporalPass::TemporalPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/asvgf/"
      "asvgf-temporal.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto TemporalPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  // Writeonly
  reflector.addOutput("MomentsHistlenHF")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RGBA16_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("ColorHistlenSpec")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RGBA16_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("AtrousHF")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::R32_UINT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("AtrousSpec")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::R32_UINT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("AtrousMoments")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RG16_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("AtrousHFBack")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::R32_UINT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("AtrousSpecBack")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::R32_UINT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("AtrousMomentsBack")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RG16_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Debug")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  // Readonly
  reflector.addInput("GradHF")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("HF")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("Spec")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("MomentsHistlenHFPrev")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RGBA16_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("ColorHistlenSpecPrev")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RGBA16_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("HFFilteredPrev")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::R32_UINT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Composite")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  // gbuffer
  GBufferUtils::addGBufferInputOutput(
      reflector, (uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT);
  GBufferUtils::addPrevGbufferInputOutput(
      reflector, (uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT);
  return reflector;
}

auto TemporalPass::execute(RDG::RenderContext* context,
                                    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* mhl = renderData.getTexture("MomentsHistlenHF");
  GFX::Texture* chs = renderData.getTexture("ColorHistlenSpec");
  GFX::Texture* ahf = renderData.getTexture("AtrousHF");
  GFX::Texture* asp = renderData.getTexture("AtrousSpec");
  GFX::Texture* amo = renderData.getTexture("AtrousMoments");
  GFX::Texture* ghf = renderData.getTexture("GradHF");
  GFX::Texture* hft = renderData.getTexture("HF");
  GFX::Texture* spt = renderData.getTexture("Spec");
  GFX::Texture* mhp = renderData.getTexture("MomentsHistlenHFPrev");
  GFX::Texture* csp = renderData.getTexture("ColorHistlenSpecPrev");
  GFX::Texture* hfp = renderData.getTexture("HFFilteredPrev");
  GFX::Texture* dbg = renderData.getTexture("Debug");
    
  // gbuffer binding
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  GBufferUtils::bindGBufferResource(this, context, renderData);
  GBufferUtils::bindPrevGBufferResource(this, context, renderData);

  updateBindings(
      context,
      std::vector<std::pair<std::string, RHI::BindingResource>>{
          {"u_MomentsHistlenHF", RHI::BindingResource{mhl->getUAV(0, 0, 1)}},
          {"u_ColorHistlenSpec", RHI::BindingResource{chs->getUAV(0, 0, 1)}},
          {"u_AtrousHF", RHI::BindingResource{ahf->getUAV(0, 0, 1)}},
          {"u_AtrousSpec", RHI::BindingResource{asp->getUAV(0, 0, 1)}},
          {"u_AtrousMoments", RHI::BindingResource{amo->getUAV(0, 0, 1)}},
          {"u_MomentsHistlenHF_Prev", RHI::BindingResource{mhp->getUAV(0, 0, 1)}},
          {"u_ColorHistlenSpec_Prev", RHI::BindingResource{csp->getUAV(0, 0, 1)}},
          {"u_GradHFSpec", RHI::BindingResource{ghf->getUAV(0, 0, 1)}},
          {"u_HF", RHI::BindingResource{hft->getUAV(0, 0, 1)}},
          {"u_Spec", RHI::BindingResource{spt->getUAV(0, 0, 1)}},
          {"u_HFFiltered_Prev", RHI::BindingResource{hfp->getUAV(0, 0, 1)}},
          {"u_Debug", RHI::BindingResource{dbg->getUAV(0, 0, 1)}},
      });
  
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroups((1280 + 14) / 15, (720 + 14) / 15, 1);
  encoder->end();
}

AtrousPass::AtrousPass(int iter) : iteration(iter) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/asvgf/"
      "asvgf-atrous.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto AtrousPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  // Writeonly
  reflector.addInputOutput("AtrousSpecPong")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("AtrousHFPong")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("AtrousMomentPong")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("Composite")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  // Readonly
  reflector.addInput("AtrousSpecPing")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("AtrousHFPing")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("AtrousMomentPing")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("MomentsHistlenHF")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RGBA16_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("IsCorrelated")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  // gbuffer
  GBufferUtils::addGBufferInputOutput(
      reflector, (uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT);
  return reflector;
}

auto AtrousPass::execute(RDG::RenderContext* context,
                                    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* hfpo = renderData.getTexture("AtrousHFPong");
  GFX::Texture* sppo = renderData.getTexture("AtrousSpecPong");
  GFX::Texture* mopo = renderData.getTexture("AtrousMomentPong");
  GFX::Texture* sppi = renderData.getTexture("AtrousSpecPing");
  GFX::Texture* hfpi = renderData.getTexture("AtrousHFPing");
  GFX::Texture* mopi = renderData.getTexture("AtrousMomentPing");
  GFX::Texture* mohl = renderData.getTexture("MomentsHistlenHF");
  GFX::Texture* compo = renderData.getTexture("Composite");
  GFX::Texture* icorr = renderData.getTexture("IsCorrelated");
    
  // gbuffer binding
  GBufferUtils::bindGBufferResource(this, context, renderData);

  updateBindings(context,
      std::vector<std::pair<std::string, RHI::BindingResource>>{
          {"u_Atrous_HF", RHI::BindingResource{hfpo->getUAV(0, 0, 1)}},
          {"u_Atrous_Spec", RHI::BindingResource{sppo->getUAV(0, 0, 1)}},
          {"u_Atrous_Moments", RHI::BindingResource{mopo->getUAV(0, 0, 1)}},
          {"s_Atrous_Spec", RHI::BindingResource{sppi->getUAV(0, 0, 1)}},
          {"s_Atrous_HF", RHI::BindingResource{hfpi->getUAV(0, 0, 1)}},
          {"s_Atrous_Moments", RHI::BindingResource{mopi->getUAV(0, 0, 1)}},
          {"u_MomentsHistlenHF", RHI::BindingResource{mohl->getUAV(0, 0, 1)}},
          {"u_Color", RHI::BindingResource{compo->getUAV(0, 0, 1)}},
          {"u_IsCorrelated", RHI::BindingResource{icorr->getUAV(0, 0, 1)}},
      });

  struct PushConstant {
    Math::ivec2 resolution;
    int spec_iteration;
    int rnd_seed;
  } pConst = {
    {1280, 720},
    iteration,
    renderData.getUInt("AccumIdx")
  };
  
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((1280 + 15) / 16, (720 + 15) / 16, 1);
  encoder->end();
}

DebugViewer::DebugViewer() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/asvgf/"
      "asvgf-viewer.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto DebugViewer::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  // Writeonly
  reflector.addOutput("Color")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  // Readonly
  reflector.addInputOutput("HF")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto DebugViewer::execute(RDG::RenderContext* context,
                                    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* hf = renderData.getTexture("HF");
  updateBindings(context,
      std::vector<std::pair<std::string, RHI::BindingResource>>{
          {"u_Color", RHI::BindingResource{color->getUAV(0, 0, 1)}},
          {"u_HF", RHI::BindingResource{hf->getUAV(0, 0, 1)}},
      });
  
  RHI::ComputePassEncoder* encoder = beginPass(context);
  //encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
  //                       sizeof(PushConstant));
  encoder->dispatchWorkgroups((1280 + 15) / 16, (720 + 15) / 16, 1);
  encoder->end();
}

auto FinaleGraph::alias() noexcept -> RDG::AliasDict {
  RDG::AliasDict dict;
  dict.addAlias("ViewDepth Source", CONCAT("Blit ViewDepth"), "Source");
  dict.addAlias("DiffuseAlbedo Source", CONCAT("Blit DiffuseAlbedo"), "Source");
  dict.addAlias("SpecularRough Source", CONCAT("Blit SpecularRough"), "Source");
  dict.addAlias("Normal Source", CONCAT("Blit Normal"), "Source");
  dict.addAlias("GeometryNormal Source", CONCAT("Blit GeometryNormal"),
                "Source");

  dict.addAlias("ViewDepth Target", CONCAT("Blit ViewDepth"), "Target");
  dict.addAlias("DiffuseAlbedo Target", CONCAT("Blit DiffuseAlbedo"), "Target");
  dict.addAlias("SpecularRough Target", CONCAT("Blit SpecularRough"), "Target");
  dict.addAlias("Normal Target", CONCAT("Blit Normal"), "Target");
  dict.addAlias("GeometryNormal Target", CONCAT("Blit GeometryNormal"),
                "Target");

  return dict;
}

auto FinaleGraph::onRegister(RDG::Graph* graph) noexcept -> void {
  graph->addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{
                     0, 0, 0, 0, BlitPass::SourceType::FLOAT}),
                 CONCAT("Blit ViewDepth"));
  graph->addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{
                     0, 0, 0, 0, BlitPass::SourceType::UINT}),
                 CONCAT("Blit DiffuseAlbedo"));
  graph->addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{
                     0, 0, 0, 0, BlitPass::SourceType::UINT}),
                 CONCAT("Blit SpecularRough"));
  graph->addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{
                     0, 0, 0, 0, BlitPass::SourceType::UINT}),
                 CONCAT("Blit Normal"));
  graph->addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{
                     0, 0, 0, 0, BlitPass::SourceType::UINT}),
                 CONCAT("Blit GeometryNormal"));
}
}