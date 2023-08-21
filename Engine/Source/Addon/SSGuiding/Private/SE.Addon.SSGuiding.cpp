#include "../Public/SE.Addon.SSGuiding.hpp"

namespace SIByL::Addon::SSGuiding {
SSPGvMF_ClearPass::SSPGvMF_ClearPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/pathguiding/"
      "sspg-vMF-clear.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto SSPGvMF_ClearPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("vMFStatistics")
      .isTexture()
      .withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("EpochCounter")
      .isTexture()
      .withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::R16_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto SSPGvMF_ClearPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* ec = renderData.getTexture("EpochCounter");
  updateBinding(context, "u_epochCounter", RHI::BindingResource{ec->getUAV(0, 0, 1)});

  struct PushConstant {
    Math::ivec2 resolution;
    int clearAll;
  } pConst;
  pConst.resolution = {1280, 720};
  pConst.clearAll = clear ? 1 : 0;
  if (clear) clear = false;

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((1280 + 15) / 16, (720 + 15) / 16, 1);
  encoder->end();
}

auto SSPGvMF_ClearPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Clear", &clear);
}

SSPGvMF_SamplePass::SSPGvMF_SamplePass() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "pathguiding/sspg-vMF-sample.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto SSPGvMF_SamplePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("vMFStatistics")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("EpochCounter")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("VBuffer")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto SSPGvMF_SamplePass::renderUI() noexcept -> void {
  ImGui::DragInt("Strategy", &strategy);
}

auto SSPGvMF_SamplePass::execute(RDG::RenderContext* context,
                           RDG::RenderData const& renderData) noexcept -> void {
  // Bind common for RT
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* ec = renderData.getTexture("EpochCounter");
  GFX::Texture* vs = renderData.getTexture("vMFStatistics");
  GFX::Texture* color = renderData.getTexture("Color");

  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{vbuffer->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_vMFStatistics",
                RHI::BindingResource{vs->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_epochCounter",
                RHI::BindingResource{ec->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_color",
                RHI::BindingResource{color->getSRV(0, 1, 0, 1)});

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  struct PushConstant {
    Math::ivec2 resolution;
    uint32_t sample_batch;
    uint32_t strategy;
  } pConst = {{1280, 720}, renderData.getUInt("AccumIdx"), strategy};

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(1280, 720, 1);
  encoder->end();
}

SSPGvMF_VisPass::SSPGvMF_VisPass() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "pathguiding/sspg-vMF-vis.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto SSPGvMF_VisPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("vMFStatistics")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("VBuffer")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Color")
      .isTexture()
      .withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto SSPGvMF_VisPass::renderUI() noexcept -> void {
  ImGui::DragInt("Debug Mode", &debugMode, 1, 0, 1);
  ImGui::DragFloat("Debug Scalar", &scalar, 0.01);
  ImGui::DragInt("Debug Pixel X", &debugPixel.x);
  ImGui::DragInt("Debug Pixel Y", &debugPixel.y);
}

auto SSPGvMF_VisPass::onInteraction(
    Platform::Input* input, Editor::Widget::WidgetInfo* info) noexcept -> void {
  if (info->isFocused && info->isHovered) {
    if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
      debugPixel.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
      debugPixel.y = std::clamp(info->mousePos.y, 0.f, 719.f);
    }
  }
}

auto SSPGvMF_VisPass::execute(RDG::RenderContext* context,
                           RDG::RenderData const& renderData) noexcept -> void {
  // Bind common for RT
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* vs = renderData.getTexture("vMFStatistics");
  GFX::Texture* color = renderData.getTexture("Color");

  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{vbuffer->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_vMFStatistics",
                RHI::BindingResource{vs->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_output",
                RHI::BindingResource{color->getSRV(0, 1, 0, 1)});

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  struct PushConstant {
    Math::ivec2 resolution;
    Math::ivec2 debugpixel;
    int mode;
    float scalar;
  } pConst = {{1280, 720}, debugPixel, debugMode, scalar};

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(512, 512, 1);
  encoder->end();
}
}