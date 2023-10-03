#include "../Public/SE.Addon.SSGuiding.hpp"
#include <SE.Addon.GBuffer.hpp>

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
  ImGui::Checkbox("Learn", &learn);
  ImGui::Checkbox("Learn one frame", &learn_one_frame);
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
    uint32_t learn;
  } pConst = {{1280, 720}, renderData.getUInt("FrameIdx"), strategy};
    pConst.learn = learn ? 1 : 0;
  if (learn_one_frame) {
    pConst.learn = 1;
    learn_one_frame = false;
  }
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
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("VBuffer")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("PdfNormalizing")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
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
 /* std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);*/

  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* vs = renderData.getTexture("vMFStatistics");
  GFX::Texture* pdf = renderData.getTexture("PdfNormalizing");

  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{vbuffer->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_vMFStatistics",
                RHI::BindingResource{vs->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_PdfNormalizing",
                RHI::BindingResource{pdf->getSRV(0, 1, 0, 1)});

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

SSPGGMM_ClearPass::SSPGGMM_ClearPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/pathguiding/"
      "sspg-gmm-clear.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto SSPGGMM_ClearPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("GMMStatisticsPack0")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("GMMStatisticsPack1")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("GMMStatisticsPack0Prev")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("GMMStatisticsPack1Prev")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto SSPGGMM_ClearPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* pack0 = renderData.getTexture("GMMStatisticsPack0");
  GFX::Texture* pack1 = renderData.getTexture("GMMStatisticsPack1");
  GFX::Texture* pack0prev = renderData.getTexture("GMMStatisticsPack0Prev");
  GFX::Texture* pack1prev = renderData.getTexture("GMMStatisticsPack1Prev");
  updateBindings(context, {
      {"u_gmmStatisticsPack0", RHI::BindingResource{pack0->getUAV(0, 0, 1)}},
      {"u_gmmStatisticsPack1", RHI::BindingResource{pack1->getUAV(0, 0, 1)}},
      {"u_gmmStatisticsPack0Prev", RHI::BindingResource{pack0prev->getUAV(0, 0, 1)}},
      {"u_gmmStatisticsPack1Prev", RHI::BindingResource{pack1prev->getUAV(0, 0, 1)}},});

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

auto SSPGGMM_ClearPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Clear", &clear);
}

SSPGGMM_SamplePass::SSPGGMM_SamplePass() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "pathguiding/sspg-gmm-sample-g.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto SSPGGMM_SamplePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("GMMStatisticsPack0")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("GMMStatisticsPack1")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("GMMStatisticsPack0Prev")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("GMMStatisticsPack1Prev")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Color")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("VPLs")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  GBufferUtils::addGBufferInput(
      reflector, (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR);
  GBufferUtils::addPrevGbufferInputOutput(
      reflector, (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR);
  return reflector;
}

auto SSPGGMM_SamplePass::renderUI() noexcept -> void {
  ImGui::DragInt("Strategy", &strategy);
  ImGui::DragInt("SPP", &spp);
  ImGui::Checkbox("Learn", &learn);
  ImGui::Checkbox("Learn one frame", &learn_one_frame);
  ImGui::Checkbox("Extra half spp", &extra_half_spp);
  ImGui::DragFloat("Exponential Factor", &expon_factor);
  ImGui::Checkbox("Multi Bounce", &multi_bounce);
  ImGui::Checkbox("Learn First", &learn_first);
}

auto SSPGGMM_SamplePass::execute(RDG::RenderContext* context,
                           RDG::RenderData const& renderData) noexcept -> void {
  // Bind common for RT
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);
  // Bind gbuffer for RT
  GBufferUtils::bindGBufferResource(this, context, renderData);
  GBufferUtils::bindPrevGBufferResource(this, context, renderData);

  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* pack0 = renderData.getTexture("GMMStatisticsPack0");
  GFX::Texture* pack1 = renderData.getTexture("GMMStatisticsPack1");
  GFX::Texture* pack0prev = renderData.getTexture("GMMStatisticsPack0Prev");
  GFX::Texture* pack1prev = renderData.getTexture("GMMStatisticsPack1Prev");
  GFX::Texture* vpls = renderData.getTexture("VPLs");

  updateBindings(context, {
      {"u_color", RHI::BindingResource{color->getSRV(0, 1, 0, 1)}},
      {"u_gmmStatisticsPack0", RHI::BindingResource{pack0->getUAV(0, 0, 1)}},
      {"u_gmmStatisticsPack1", RHI::BindingResource{pack1->getUAV(0, 0, 1)}},
      {"u_gmmStatisticsPack0Prev", RHI::BindingResource{pack0prev->getUAV(0, 0, 1)}},
      {"u_gmmStatisticsPack1Prev", RHI::BindingResource{pack1prev->getUAV(0, 0, 1)}},
      {"u_vpls", RHI::BindingResource{vpls->getUAV(0, 0, 1)}},
      {"PrevGlobalUniforms", renderData.getBindingResource("PrevGlobalUniforms").value()},
  });

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  struct PushConstant {
    Math::ivec2 resolution;
    uint32_t sample_batch;
    uint32_t strategy;
    uint32_t learn;
    float expon;
    uint32_t multi_bounce;
    int spp;
    int extra_half_spp;
    int learn_first;
  } pConst = {
      {1280, 720}, renderData.getUInt("FrameIdx"), strategy, 0, expon_factor};
  pConst.learn = learn ? 1 : 0;
  pConst.spp = spp;
  pConst.multi_bounce = multi_bounce ? 1 : 0;
  pConst.extra_half_spp = extra_half_spp ? 1 : 0;
  pConst.learn_first = learn_first ? 1 : 0;
  if (learn_one_frame) {
    pConst.learn = 1;
    learn_one_frame = false;
  }

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(1280, 720, 1);
  encoder->end();
}

SSPGGMM_LearnPass::SSPGGMM_LearnPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/pathguiding/"
      "sspg-gmm-learning.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto SSPGGMM_LearnPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("GMMStatisticsPack0")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("GMMStatisticsPack1")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("VPLs")
      .isTexture().withSize(Math::vec3(1.f))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  GBufferUtils::addGBufferInput(
      reflector, (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR);
  return reflector;
}

auto SSPGGMM_LearnPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Learn", &learn);
  ImGui::Checkbox("Learn one frame", &learn_one_frame);
  ImGui::DragInt("Extra sample", &extra_sample);
}

auto SSPGGMM_LearnPass::execute(RDG::RenderContext* context,
                           RDG::RenderData const& renderData) noexcept -> void {
  // Bind common for RT
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  // Bind gbuffer for RT
  GBufferUtils::bindGBufferResource(this, context, renderData);
    
  GFX::Texture* pack0 = renderData.getTexture("GMMStatisticsPack0");
  GFX::Texture* pack1 = renderData.getTexture("GMMStatisticsPack1");
  GFX::Texture* vpls = renderData.getTexture("VPLs");

  updateBindings(context, {
      {"u_gmmStatisticsPack0", RHI::BindingResource{pack0->getUAV(0, 0, 1)}},
      {"u_gmmStatisticsPack1", RHI::BindingResource{pack1->getUAV(0, 0, 1)}},
      {"u_vpls", RHI::BindingResource{vpls->getUAV(0, 0, 1)}},
  });

  struct PushConstant {
    Math::ivec2 resolution;
    int reuse_number;
    float exponential_factor;
    int learn;
    int random_seed;
  } pConst = {{1280, 720}, extra_sample, 0.7, renderData.getUInt("FrameIdx")
  };
  pConst.learn = learn ? 1 : 0;
  if (learn_one_frame) {
    pConst.learn = 1;
    learn_one_frame = false;
  }
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((1280 + 15) / 16, (720 + 15) / 16, 1);
  encoder->end();
}

SSPGGMM_VisPass::SSPGGMM_VisPass() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "pathguiding/sspg-gmm-vis.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto SSPGGMM_VisPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("GMMStatisticsPack0")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("GMMStatisticsPack1")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("VBuffer")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Color")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("PdfNormalizing")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto SSPGGMM_VisPass::renderUI() noexcept -> void {
  ImGui::DragInt("Debug Mode", &debugMode, 1, 0, 1);
  ImGui::DragFloat("Debug Scalar", &scalar, 0.01);
  ImGui::DragInt("Debug Pixel X", &debugPixel.x);
  ImGui::DragInt("Debug Pixel Y", &debugPixel.y);
}

auto SSPGGMM_VisPass::onInteraction(
    Platform::Input* input, Editor::Widget::WidgetInfo* info) noexcept -> void {
  if (info->isFocused && info->isHovered) {
    if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
      debugPixel.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
      debugPixel.y = std::clamp(info->mousePos.y, 0.f, 719.f);
    }
  }
}

auto SSPGGMM_VisPass::execute(RDG::RenderContext* context,
                           RDG::RenderData const& renderData) noexcept -> void {
  // Bind common for RT
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* pack0 = renderData.getTexture("GMMStatisticsPack0");
  GFX::Texture* pack1 = renderData.getTexture("GMMStatisticsPack1");
  GFX::Texture* pdfn = renderData.getTexture("PdfNormalizing");

  updateBindings(context, {
      {"u_gmmStatisticsPack0", RHI::BindingResource{pack0->getUAV(0, 0, 1)}},
      {"u_gmmStatisticsPack1", RHI::BindingResource{pack1->getUAV(0, 0, 1)}},
      {"u_pdfUnormalized", RHI::BindingResource{pdfn->getUAV(0, 0, 1)}},});
  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{vbuffer->getSRV(0, 1, 0, 1)});
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

SSPGGMM_TestPass::SSPGGMM_TestPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/pathguiding/"
      "sspg-gmm-test.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto SSPGGMM_TestPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("PdfAccumulator")
      .isTexture().withSize(Math::ivec3(512,512,1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("PdfAccumulatorInfo")
      .isTexture().withSize(Math::ivec3(2,2,1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("GMMStatisticsPack0")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("GMMStatisticsPack1")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto SSPGGMM_TestPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* pdf_accum = renderData.getTexture("PdfAccumulator");
  GFX::Texture* pdf_ainfo = renderData.getTexture("PdfAccumulatorInfo");
  updateBindings(context, {
      {"u_PdfAccumulator", RHI::BindingResource{pdf_accum->getUAV(0, 0, 1)}},
      {"u_PdfAccumulatorInfo", RHI::BindingResource{pdf_ainfo->getUAV(0, 0, 1)}}});
  GFX::Texture* pack0 = renderData.getTexture("GMMStatisticsPack0");
  GFX::Texture* pack1 = renderData.getTexture("GMMStatisticsPack1");
  updateBindings(context, {
      {"u_gmmStatisticsPack0", RHI::BindingResource{pack0->getUAV(0, 0, 1)}},
      {"u_gmmStatisticsPack1", RHI::BindingResource{pack1->getUAV(0, 0, 1)}}});

  struct PushConstant {
    Math::ivec2 resolution;
    Math::ivec2 debug_pixel;
    int rand_seed;
  } pConst;
  pConst.resolution = {512,512};
  pConst.debug_pixel = debugPixel;
  pConst.rand_seed = renderData.getUInt("FrameIdx");
    
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((512 + 15) / 16, (512 + 15) / 16, 1);
  encoder->end();
}

auto SSPGGMM_TestPass::onInteraction(
    Platform::Input* input, Editor::Widget::WidgetInfo* info) noexcept -> void {
  if (info->isFocused && info->isHovered) {
    if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
      debugPixel.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
      debugPixel.y = std::clamp(info->mousePos.y, 0.f, 719.f);
    }
  }
}

PdfAccum_ClearPass::PdfAccum_ClearPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/pathguiding/"
      "pdf-accumulation/pdf-accumed-clear.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto PdfAccum_ClearPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("PdfAccumulator")
      .isTexture().withSize(Math::ivec3(512,512,1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("PdfAccumulatorInfo")
      .isTexture().withSize(Math::ivec3(2,2,1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto PdfAccum_ClearPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* pdf_accum = renderData.getTexture("PdfAccumulator");
  GFX::Texture* pdf_ainfo = renderData.getTexture("PdfAccumulatorInfo");
  updateBindings(context, {
      {"u_PdfAccumulator", RHI::BindingResource{pdf_accum->getUAV(0, 0, 1)}},
      {"u_PdfAccumulatorInfo", RHI::BindingResource{pdf_ainfo->getUAV(0, 0, 1)}}});

  struct PushConstant {
    Math::ivec2 resolution;
    int clearAll;
  } pConst;
  pConst.resolution = {512,512};
  pConst.clearAll = clear ? 1 : 0;
  if (clear) clear = false;

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((512 + 15) / 16, (512 + 15) / 16, 1);
  encoder->end();
}

auto PdfAccum_ClearPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Clear", &clear);
}

PdfAccum_ViewerPass::PdfAccum_ViewerPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/pathguiding/"
      "pdf-accumulation/pdf-accumed-viewer.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto PdfAccum_ViewerPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("PdfAccumulator")
      .isTexture().withSize(Math::ivec3(512,512,1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("PdfAccumulatorInfo")
      .isTexture().withSize(Math::ivec3(2,2,1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Color")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::RGBA8_UNORM)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Importance")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto PdfAccum_ViewerPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* pdf_accum = renderData.getTexture("PdfAccumulator");
  GFX::Texture* pdf_ainfo = renderData.getTexture("PdfAccumulatorInfo");
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* importance = renderData.getTexture("Importance");
  updateBindings(context, {
      {"s_PdfAccumulator", RHI::BindingResource{pdf_accum->getUAV(0, 0, 1)}},
      {"s_PdfAccumulatorInfo", RHI::BindingResource{pdf_ainfo->getUAV(0, 0, 1)}},
      {"u_Importance", RHI::BindingResource{importance->getUAV(0, 0, 1)}},
      {"u_Output", RHI::BindingResource{color->getUAV(0, 0, 1)}}});

  struct PushConstant {
    Math::ivec2 resolution;
    float scalar;
  } pConst;
  pConst.resolution = {512,512};
  pConst.scalar = scalar;

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((512 + 15) / 16, (512 + 15) / 16, 1);
  encoder->end();
}

auto PdfAccum_ViewerPass::renderUI() noexcept -> void {
  ImGui::DragFloat("Scalar", &scalar);
}

PdfNormalize_ClearPass::PdfNormalize_ClearPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/pathguiding/"
      "pdf-normalizing/pdf-normalizing-clear.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto PdfNormalize_ClearPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("PdfNormalizing")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("PdfNormalizingInfo")
      .isTexture().withSize(Math::ivec3(2, 2, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto PdfNormalize_ClearPass::execute(RDG::RenderContext* context,
                                 RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* pdf_norml = renderData.getTexture("PdfNormalizing");
  GFX::Texture* pdf_ninfo = renderData.getTexture("PdfNormalizingInfo");
  updateBindings(context, {
      {"u_PdfNormalizing", RHI::BindingResource{pdf_norml->getUAV(0, 0, 1)}},
      {"u_PdfNormalizingInfo", RHI::BindingResource{pdf_ninfo->getUAV(0, 0, 1)}}});

  struct PushConstant {
    Math::ivec2 resolution;
    int clearAll;
  } pConst;
  pConst.resolution = {512, 512};
  pConst.clearAll = clear ? 1 : 0;
  
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((512 + 15) / 16, (512 + 15) / 16, 1);
  encoder->end();
}

auto PdfNormalize_ClearPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Clear", &clear);
}

PdfNormalize_SumPass::PdfNormalize_SumPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/pathguiding/"
      "pdf-normalizing/pdf-normalizing-sum.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto PdfNormalize_SumPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("PdfNormalizing")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("PdfNormalizingInfo")
      .isTexture().withSize(Math::ivec3(2, 2, 1))
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto PdfNormalize_SumPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* pdf_norml = renderData.getTexture("PdfNormalizing");
  GFX::Texture* pdf_ninfo = renderData.getTexture("PdfNormalizingInfo");
  updateBindings(context, {
      {"s_PdfNormalizing", RHI::BindingResource{pdf_norml->getUAV(0, 0, 1)}},
      {"u_PdfNormalizingInfo", RHI::BindingResource{pdf_ninfo->getUAV(0, 0, 1)}}});

  struct PushConstant {
    Math::ivec2 resolution;
  } pConst;
  pConst.resolution = {512,512};
  
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((512 + 15) / 16, (512 + 15) / 16, 1);
  encoder->end();
}

PdfNormalize_ViewerPass::PdfNormalize_ViewerPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/pathguiding/"
      "pdf-normalizing/pdf-normalizing-viewer.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto PdfNormalize_ViewerPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("PdfNormalizing")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("PdfNormalizingInfo")
      .isTexture().withSize(Math::ivec3(2, 2, 1))
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Color")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::RGBA8_UNORM)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Importance")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto PdfNormalize_ViewerPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* pdf_accum = renderData.getTexture("PdfNormalizing");
  GFX::Texture* pdf_ainfo = renderData.getTexture("PdfNormalizingInfo");
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* importance = renderData.getTexture("Importance");
  updateBindings(context, {
      {"s_PdfNormalizing", RHI::BindingResource{pdf_accum->getUAV(0, 0, 1)}},
      {"s_PdfNormalizingInfo", RHI::BindingResource{pdf_ainfo->getUAV(0, 0, 1)}},
      {"u_Irradiance", RHI::BindingResource{importance->getUAV(0, 0, 1)}},
      {"u_Output", RHI::BindingResource{color->getUAV(0, 0, 1)}}});

  struct PushConstant {
    Math::ivec2 resolution;
    float scalar;
  } pConst;
  pConst.resolution = {512,512};
  pConst.scalar = scalar;

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((512 + 15) / 16, (512 + 15) / 16, 1);
  encoder->end();
}

auto PdfNormalize_ViewerPass::renderUI() noexcept -> void {
  ImGui::DragFloat("Scalar", &scalar, 0.01);
}

PdfNormalize_TestPass::PdfNormalize_TestPass() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "pathguiding/pdf-normalizing/pdf-normalizing-test.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto PdfNormalize_TestPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("VBuffer")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("PdfNormalizing")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto PdfNormalize_TestPass::renderUI() noexcept -> void {
  ImGui::DragInt("Debug Pixel X", &debugPixel.x);
  ImGui::DragInt("Debug Pixel Y", &debugPixel.y);
}

auto PdfNormalize_TestPass::onInteraction(
    Platform::Input* input, Editor::Widget::WidgetInfo* info) noexcept -> void {
  if (info->isFocused && info->isHovered) {
    if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
      debugPixel.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
      debugPixel.y = std::clamp(info->mousePos.y, 0.f, 719.f);
    }
  }
}

auto PdfNormalize_TestPass::execute(RDG::RenderContext* context,
                           RDG::RenderData const& renderData) noexcept -> void {
  // Bind common for RT
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* pdf = renderData.getTexture("PdfNormalizing");

  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{vbuffer->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_PdfNormalizing",
                RHI::BindingResource{pdf->getSRV(0, 1, 0, 1)});

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  struct PushConstant {
    Math::ivec2 resolution;
    Math::ivec2 debugpixel;
  } pConst = {{1280, 720}, debugPixel};

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(512, 512, 1);
  encoder->end();
}

CDQ_PresamplePass::CDQ_PresamplePass() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "pathguiding/cdq/cdq-sampling.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto CDQ_PresamplePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("VBuffer")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("PresampleList")
      .isTexture().withSize(Math::ivec3(16, 16, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto CDQ_PresamplePass::renderUI() noexcept -> void {
  ImGui::DragInt("Debug Pixel X", &debugPixel.x);
  ImGui::DragInt("Debug Pixel Y", &debugPixel.y);
}

auto CDQ_PresamplePass::onInteraction(
    Platform::Input* input, Editor::Widget::WidgetInfo* info) noexcept -> void {
  if (info->isFocused && info->isHovered) {
    if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
      debugPixel.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
      debugPixel.y = std::clamp(info->mousePos.y, 0.f, 719.f);
    }
  }
}

auto CDQ_PresamplePass::execute(RDG::RenderContext* context,
                                    RDG::RenderData const& renderData) noexcept
    -> void {
  // Bind common for RT
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* presample = renderData.getTexture("PresampleList");

  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{vbuffer->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_PreSampleList",
                RHI::BindingResource{presample->getSRV(0, 1, 0, 1)});

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  struct PushConstant {
    Math::ivec2 resolution;
    Math::ivec2 debugpixel;
    int rand_seed;
  } pConst = {{1280, 720}, debugPixel, renderData.getUInt("FrameIdx")};

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(16, 16, 1);
  encoder->end();
}

CDQ_AdaptionPass::CDQ_AdaptionPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/pathguiding/"
      "cdq/cdq-adaption.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto CDQ_AdaptionPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("CDQBuffer")
      .isBuffer().withSize(128 * sizeof(uint64_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("PresampleList")
      .isTexture().withSize(Math::ivec3(16, 16, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto CDQ_AdaptionPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* cdq = renderData.getBuffer("CDQBuffer");
  GFX::Texture* presample = renderData.getTexture("PresampleList");

  updateBinding(context, "u_PreSampleList",
                RHI::BindingResource{presample->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_cdq",
      RHI::BindingResource{{cdq->buffer.get(), 0, cdq->buffer->size()}});

  RHI::ComputePassEncoder* encoder = beginPass(context);
  struct PushConstant {
    int initialize = 0;
    int learn = 0;
  } pConst;
  if (initialize) {
    pConst.initialize = 1;
    initialize = false;
  }
  pConst.learn = learn ? 1 : 0;
  if (learn_one_frame) {
    pConst.learn = 1;
    learn_one_frame = false;
  }
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups(1, 1, 1);
  encoder->end();
}

auto CDQ_AdaptionPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Initialize", &initialize);
  ImGui::Checkbox("Learn", &learn);
  ImGui::Checkbox("Learn one frame", &learn_one_frame);
}

CDQ_VisualizePass::CDQ_VisualizePass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/pathguiding/"
      "cdq/cdq-visualize.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto CDQ_VisualizePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("PdfNormalizing")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("CDQBuffer")
      .isBuffer().withSize(128 * sizeof(uint64_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto CDQ_VisualizePass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* cdq = renderData.getBuffer("CDQBuffer");
  GFX::Texture* pdf_accum = renderData.getTexture("PdfNormalizing");
  updateBindings(context, {
      {"u_PdfNormalizing", RHI::BindingResource{pdf_accum->getUAV(0, 0, 1)}},
      {"u_cdq", RHI::BindingResource{{cdq->buffer.get(), 0, cdq->buffer->size()}}},
      });

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroups((512 + 15) / 16, (512 + 15) / 16, 1);
  encoder->end();
}
}