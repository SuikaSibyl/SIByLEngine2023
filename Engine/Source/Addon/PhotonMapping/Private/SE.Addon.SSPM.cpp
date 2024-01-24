#include "../Public/SE.Addon.SSPM.hpp"
#include <SE.Addon.VBuffer.hpp>

namespace SIByL::Addon::SSPM {
SSPMClearPass::SSPMClearPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
    "../Engine/Shaders/SRenderer/addon/photon/sspm-clear.slang",
    std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
      std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),});
  RDG::ComputePass::init(
    Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto SSPMClearPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("VBuffer")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
      RDG::TextureInfo::ConsumeType::StorageBinding}
        .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Depth")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RG32_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
      RDG::TextureInfo::ConsumeType::StorageBinding}
        .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("PhotonMap")
    .isBuffer().withSize(1280 * 720 * sizeof(Math::vec3))
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto SSPMClearPass::execute(RDG::RenderContext* context,
                            RDG::RenderData const& renderData) noexcept -> void {
  std::vector<RHI::BindGroupEntry>* set_0_entries =
    renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  
  GFX::Texture* vb = renderData.getTexture("VBuffer");
  GFX::Texture* depth = renderData.getTexture("Depth");
  GFX::Buffer* photonmap = renderData.getBuffer("PhotonMap");

  updateBindings(context, {
    {"u_vBuffer", RHI::BindingResource{vb->getSRV(0, 1, 0, 1)}},
    {"u_depth", RHI::BindingResource{depth->getUAV(0, 0, 1)}},
    {"u_photon_map", RHI::BindingResource{{photonmap->buffer.get(), 0, photonmap->buffer->size()}}},
  });

  struct PushConstant {
    Math::ivec2 resolution;
    //int random_seed;
  } pConst = {{1280, 720}};

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((1280 + 15) / 16, (720 + 15) / 16, 1);
  encoder->end();
}

SSPMPass::SSPMPass() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/photon/sspm-tracing.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),});
  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};
  RayTracingPass::init(sbt, 1);
}

auto SSPMPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("Depth")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
      RDG::TextureInfo::ConsumeType::StorageBinding}
        .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("PhotonMap")
    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT
                |(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto SSPMPass::renderUI() noexcept -> void {
}

auto SSPMPass::execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void{
  GFX::Texture* depth = renderData.getTexture("Depth");
  GFX::Buffer* photonmap = renderData.getBuffer("PhotonMap");
  
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);
  
  updateBindings(context, {
    {"u_depth", RHI::BindingResource{depth->getUAV(0, 0, 1)}},
    {"u_photon_map", RHI::BindingResource{{photonmap->buffer.get(), 0, photonmap->buffer->size()}}},
  });

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("FrameIdx");

  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t photon_map_width;
    uint32_t photon_map_height;
    float roughness_threshold;
    float intenisty_threshold;
    int max_iteration_count;
    uint32_t rand_seed;
  };
  PushConstant pConst = {1280, 720, 512, 512, 0.2f, 0.0f, 5, renderData.getUInt("FrameIdx")};
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0, sizeof(PushConstant));
  encoder->traceRays(512, 512, 1);
  encoder->end();
}

SSPMViewPass::SSPMViewPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
    "../Engine/Shaders/SRenderer/addon/photon/sspm-view.slang",
    std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
      std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),});
  RDG::ComputePass::init(
    Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto SSPMViewPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("PhotonMap")
    .isBuffer().withSize(1280 * 720 * sizeof(Math::vec3))
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Output")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
      RDG::TextureInfo::ConsumeType::StorageBinding}
        .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto SSPMViewPass::execute(RDG::RenderContext* context,
                            RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* output = renderData.getTexture("Output");
  GFX::Buffer* photonmap = renderData.getBuffer("PhotonMap");

  updateBindings(context, {
    {"u_color", RHI::BindingResource{output->getUAV(0, 0, 1)}},
    {"u_photon_map", RHI::BindingResource{{photonmap->buffer.get(), 0, photonmap->buffer->size()}}},
  });

  struct PushConstant {
    Math::ivec2 resolution;
  } pConst = {{1280, 720}};

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(PushConstant));
  encoder->dispatchWorkgroups((1280 + 15) / 16, (720 + 15) / 16, 1);
  encoder->end();
}

SSPMGraph::SSPMGraph() {
  // Create VBuffer + GBuffer, as well as history GBuffer, A-SVGF buffers
  addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(), "VBuffer Pass");
  addPass(std::make_unique<Addon::SSPM::SSPMClearPass>(), "SSPMClear Pass");
  addEdge("VBuffer Pass", "VBuffer", "SSPMClear Pass", "VBuffer");
  addPass(std::make_unique<Addon::SSPM::SSPMPass>(), "SSPM Pass");
  addEdge("SSPMClear Pass", "Depth", "SSPM Pass", "Depth");
  addEdge("SSPMClear Pass", "PhotonMap", "SSPM Pass", "PhotonMap");
  addPass(std::make_unique<Addon::SSPM::SSPMViewPass>(), "SSPMView Pass");
  addEdge("SSPM Pass", "PhotonMap", "SSPMView Pass", "PhotonMap");
  
  markOutput("SSPMView Pass", "Output");
}
}