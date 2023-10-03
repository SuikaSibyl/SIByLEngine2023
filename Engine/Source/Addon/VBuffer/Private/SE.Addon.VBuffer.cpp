#include "../Public/SE.Addon.VBuffer.hpp"
#include <SE.RHI.Profiler.hpp>

namespace SIByL::Addon::VBuffer {
RayTraceVBuffer::RayTraceVBuffer() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vbuffer/raytraced-vbuffer.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto RayTraceVBuffer::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("VBuffer")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto RayTraceVBuffer::execute(RDG::RenderContext* context,
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
  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{{vbuffer->getUAV(0, 0, 1)}});

  Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
      context->cmdEncoder, RHI::PipelineStages::TOP_OF_PIPE_BIT,
      "vbuffer");

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  struct PushConstant {
    Math::uvec2 resolution;
    uint32_t sample_batch;
  } pConst = {{vbuffer->texture->width(), vbuffer->texture->height()},
              renderData.getUInt("AccumIdx")};

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(pConst.resolution.x, pConst.resolution.y, 1);

  encoder->end();

  Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
      context->cmdEncoder, RHI::PipelineStages::BOTTOM_OF_PIPE_BIT,
      "vbuffer");
}

VBuffer2GBufferPass::VBuffer2GBufferPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vbuffer/"
      "vbuffer-2-gbuffer.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VBuffer2GBufferPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  // Writeonly
  reflector.addOutput("Position")
      .isTexture().withSize(Math::vec3(1,1,1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("DiffuseAlbedo")
      .isTexture().withSize(Math::vec3(1,1,1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("SpecularRough")
      .isTexture().withSize(Math::vec3(1,1,1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Normal")
      .isTexture().withSize(Math::vec3(1,1,1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("GeometryNormal")
      .isTexture().withSize(Math::vec3(1,1,1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Emissive")
      .isTexture().withSize(Math::vec3(1,1,1))
      .withFormat(RHI::TextureFormat::RGBA16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("MotionVector")
      .isTexture().withSize(Math::vec3(1,1,1))
      .withFormat(RHI::TextureFormat::RGBA16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("MaterialInfo")
      .isTexture().withSize(Math::vec3(1,1,1))
      .withFormat(RHI::TextureFormat::RGBA16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  // Readonly
  reflector.addInputOutput("VBuffer")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VBuffer2GBufferPass::execute(RDG::RenderContext* context,
                                   RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* position = renderData.getTexture("Position");
  GFX::Texture* diffuseAlbedo = renderData.getTexture("DiffuseAlbedo");
  GFX::Texture* specularRough = renderData.getTexture("SpecularRough");
  GFX::Texture* normal = renderData.getTexture("Normal");
  GFX::Texture* geometryNormal = renderData.getTexture("GeometryNormal");
  GFX::Texture* emissive = renderData.getTexture("Emissive");
  GFX::Texture* motionVector = renderData.getTexture("MotionVector");
  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* materialInfo = renderData.getTexture("MaterialInfo");

  // Bind common for RT
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  
  // Bind textures
  updateBindings(
      context,
      std::vector<std::pair<std::string, RHI::BindingResource>>{
          {"u_vBuffer", RHI::BindingResource{vbuffer->getUAV(0, 0, 1)}},
          {"u_GBufferPosition", RHI::BindingResource{position->getUAV(0, 0, 1)}},
          {"u_GBufferNormals", RHI::BindingResource{normal->getUAV(0, 0, 1)}},
          {"u_GBufferGeoNormals", RHI::BindingResource{geometryNormal->getUAV(0, 0, 1)}},
          {"u_GBufferDiffuseAlbedo", RHI::BindingResource{diffuseAlbedo->getUAV(0, 0, 1)}},
          {"u_GBufferSpecularRough", RHI::BindingResource{specularRough->getUAV(0, 0, 1)}},
          {"u_Emissive", RHI::BindingResource{emissive->getUAV(0, 0, 1)}},
          {"u_MotionVectors", RHI::BindingResource{motionVector->getUAV(0, 0, 1)}},
          {"u_Material", RHI::BindingResource{materialInfo->getUAV(0, 0, 1)}},
      });

  updateBindings(
      context,
      std::vector<std::pair<std::string, RHI::BindingResource>>{
          {"PrevGlobalUniforms", renderData.getBindingResource("PrevGlobalUniforms").value()},
          {"prevGeometries", renderData.getBindingResource("PrevGeometryBuffer").value()},
      });

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroups((1280 + 15) / 16, (720 + 15) / 16, 1);
  encoder->end();
}

TestVBuffer::TestVBuffer() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vbuffer/test-vbuffer.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto TestVBuffer::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("VBuffer")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto TestVBuffer::execute(RDG::RenderContext* context,
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
  GFX::Texture* color = renderData.getTexture("Color");
  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{{vbuffer->getUAV(0, 0, 1)}});
  updateBinding(context, "u_color",
                RHI::BindingResource{{color->getUAV(0, 0, 1)}});

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  struct PushConstant {
    Math::uvec2 resolution;
    uint32_t sample_batch;
  } pConst = {{vbuffer->texture->width(), vbuffer->texture->height()},
              renderData.getUInt("AccumIdx")};

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(pConst.resolution.x, pConst.resolution.y, 1);

  encoder->end();
}
}