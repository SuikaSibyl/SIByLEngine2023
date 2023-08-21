#include "../Public/SE.Addon.ASVGF.hpp"

namespace SIByL::Addon::ASVGF {
GradientReprojection::GradientReprojection() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/gSLICr/"
      "asvgf-gradient-reproject.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto GradientReprojection::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  // Writeonly
  reflector.addInputOutput("GradSamplePos")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("HfSpecLumPrev")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("VBuffer")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  // Readonly
  reflector.addInputOutput("GradSamplePosPrev")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("HFPrev")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("SpecPrev")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("VBufferPrev")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
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

  GFX::Texture* gsp_prev = renderData.getTexture("GradSamplePosPrev");
  GFX::Texture* hf_prev = renderData.getTexture("HFPrev");
  GFX::Texture* spec_prev = renderData.getTexture("SpecPrev");
  GFX::Texture* vbuffer_prev = renderData.getTexture("VBufferPrev");

  updateBindings(
      context,
      std::vector<std::pair<std::string, RHI::BindingResource>>{
          {"u_GradSamplePos", RHI::BindingResource{gsp->getUAV(0, 0, 1)}},
          {"u_HfSpecLum_prev", RHI::BindingResource{hf_spec_lum_prev->getUAV(0, 0, 1)}},
          {"u_vBuffer", RHI::BindingResource{vbuffer->getUAV(0, 0, 1)}},
          {"u_GradSamplePos_prev", RHI::BindingResource{gsp_prev->getUAV(0, 0, 1)}},
          {"u_HF_prev", RHI::BindingResource{hf_prev->getUAV(0, 0, 1)}},
          {"u_Spec_prev", RHI::BindingResource{spec_prev->getUAV(0, 0, 1)}},
          {"u_vBuffer_prev", RHI::BindingResource{vbuffer_prev->getUAV(0, 0, 1)}},
      });

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroups((1280 + 23) / 24, (720 + 23) / 24, 1);
  encoder->end();
}


}