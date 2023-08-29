#include "../Public/SE.Addon.RestirGI.hpp"
#include <SE.Addon.GBuffer.hpp>

namespace SIByL::Addon::RestirGI {
InitialSample::InitialSample(GIResamplingRuntimeParameters* param)
    : param(param) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "restirgi/gi-initial-sample.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};
  RayTracingPass::init(sbt, 1);
}

struct PackedGIReservoir {
  Math::vec3 position;
  uint32_t packed_miscData_age_M;  // See Reservoir.hlsli about the detail of the bit field.
  uint32_t packed_radiance;        // Stored as 32bit LogLUV format.
  float    weight;
  uint32_t packed_normal;  // Stored as 2x 16-bit snorms in the octahedral mapping
  float    unused;
};

auto InitialSample::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("GIReservoir")
      .isBuffer().withSize(sizeof(PackedGIReservoir) * (1280 * 720 * 2))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
    reflector.addOutput("DebugImage")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  GBufferUtils::addGBufferInput(
      reflector, (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR);
  return reflector;
}

// 32 bit Jenkins hash
static uint32_t JenkinsHash(uint32_t a) {
  // http://burtleburtle.net/bob/hash/integer.html
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

auto InitialSample::execute(RDG::RenderContext* context,
                            RDG::RenderData const& renderData) noexcept
    -> void {
  // Bind common for RT
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);
  GBufferUtils::bindGBufferResource(this, context, renderData);

  GFX::Buffer* reservoir = renderData.getBuffer("GIReservoir");
  updateBinding(context, "u_GIReservoirs",
                RHI::BindingResource{{reservoir->buffer.get(), 0, reservoir->buffer->size()}});
  GFX::Texture* debug = renderData.getTexture("DebugImage");
  updateBinding(context, "u_debug",
                RHI::BindingResource{{debug->getUAV(0, 0, 1)}});

  const uint32_t frameID = renderData.getUInt("FrameIdx");
  // update params
  { param->uniformRandomNumber = JenkinsHash(frameID);
    param->neighborOffsetMask = 8192 - 1;
  }
  struct PushConstant {
    GIResamplingRuntimeParameters params;
    uint32_t initialOutputBufferIndex;
    uint32_t sample_batch;
  } pConst = {*param, 0, frameID};

  RHI::RayTracingPassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(1280, 720, 1);
  encoder->end();
}

TemporalResampling::TemporalResampling(GIResamplingRuntimeParameters* param)
    : param(param) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "restirgi/gi-temporal-resample.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto TemporalResampling::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("GIReservoir")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("DebugImage")
      .isTexture().withSize(Math::vec3(1, 1, 1))
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

auto TemporalResampling::execute(RDG::RenderContext* context,
                           RDG::RenderData const& renderData) noexcept
    -> void {
  // Bind common for RT
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  //getBindGroup(context, 1)->updateBinding(*set_1_entries);
  GBufferUtils::bindGBufferResource(this, context, renderData);
  GBufferUtils::bindPrevGBufferResource(this, context, renderData);

  GFX::Buffer* reservoir = renderData.getBuffer("GIReservoir");
  GFX::Texture* debug = renderData.getTexture("DebugImage");
  updateBinding(context, "u_GIReservoirs",
                RHI::BindingResource{{reservoir->buffer.get(), 0, reservoir->buffer->size()}});
  updateBinding(context, "PrevGlobalUniforms",
                renderData.getBindingResource("PrevGlobalUniforms").value());
  updateBinding(context, "PrevSceneBVH",
                renderData.getBindingResource("PrevSceneBVH").value());
  updateBinding(context, "u_debug",
                RHI::BindingResource{{debug->getUAV(0, 0, 1)}});

  uint32_t frameID = renderData.getUInt("FrameIdx");
  struct PushConstant {
    GIResamplingRuntimeParameters params;
    uint32_t temporalInputBufferIndex;
    uint32_t temporalOutputBufferIndex;
    uint32_t maxHistoryLength;
    uint32_t temporalBiasCorrection;
    float temporalDepthThreshold;
    float temporalNormalThreshold;
    uint32_t enablePermutationSampling;
    uint32_t enableFallbackSampling;
    uint32_t giReservoirMaxAge;
    uint32_t sample_batch;
  } pConst = {*param,
              1,
              0,
              8,
              0,
              0.1f,
              0.5f,
              1,
              1,
              30,
              frameID};

  RHI::RayTracingPassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(1280, 720, 1);
  encoder->end();
}

void FillNeighborOffsetBuffer(uint8_t* buffer, uint32_t neighborOffsetCount) {
  int R = 250;
  const float phi2 = 1.0f / 1.3247179572447f;
  uint32_t num = 0;
  float u = 0.5f;
  float v = 0.5f;
  while (num < neighborOffsetCount * 2) {
    u += phi2;
    v += phi2 * phi2;
    if (u >= 1.0f) u -= 1.0f;
    if (v >= 1.0f) v -= 1.0f;

    float rSq = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
    if (rSq > 0.25f) continue;

    buffer[num++] = int8_t((u - 0.5f) * R);
    buffer[num++] = int8_t((v - 0.5f) * R);
  }
}

GFX::Buffer* SpatialResampling::neighbor_buffer = nullptr;

auto SpatialResampling::initNeighborOffsetBuffer() noexcept -> void {
  std::vector<uint8_t> offsets(8192 * 2);
  FillNeighborOffsetBuffer(offsets.data(), 8192);
  Core::GUID guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
  GFX::GFXManager::get()->registerBufferResource(
      guid, offsets.data(), offsets.size() * sizeof(uint8_t),
      (uint32_t)RHI::BufferUsage::STORAGE);
  neighbor_buffer = Core::ResourceManager::get()->getResource<GFX::Buffer>(guid);
}

SpatialResampling::SpatialResampling(GIResamplingRuntimeParameters* param)
    : param(param) {
  if (neighbor_buffer == nullptr) initNeighborOffsetBuffer();

  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "restirgi/gi-spatial-resampling.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto SpatialResampling::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("GIReservoir")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
reflector.addOutput("DebugImage")
    .isTexture().withSize(Math::vec3(1, 1, 1))
    .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

  GBufferUtils::addGBufferInput(
      reflector, (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR);
  return reflector;
}

auto SpatialResampling::execute(RDG::RenderContext* context,
                                 RDG::RenderData const& renderData) noexcept
    -> void {
  // Bind common for RT
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
   getBindGroup(context, 1)->updateBinding(*set_1_entries);
  GBufferUtils::bindGBufferResource(this, context, renderData);

  GFX::Buffer* reservoir = renderData.getBuffer("GIReservoir");
  updateBinding(context, "u_GIReservoirs",
                RHI::BindingResource{
                    {reservoir->buffer.get(), 0, reservoir->buffer->size()}});
  updateBinding(context, "u_NeighborOffsets",
                RHI::BindingResource{
                    {neighbor_buffer->buffer.get(), 0, neighbor_buffer->buffer->size()}});
  GFX::Texture* debug = renderData.getTexture("DebugImage");
  updateBinding(context, "u_debug",
                RHI::BindingResource{{debug->getUAV(0, 0, 1)}});

  uint32_t frameID = renderData.getUInt("FrameIdx");
  struct PushConstant {
    GIResamplingRuntimeParameters params;
    uint32_t spatialInputBufferIndex;
    uint32_t spatialOutputBufferIndex;
    uint32_t sample_batch;
    float spatialDepthThreshold;
    float spatialNormalThreshold;
    uint32_t numSpatialSamples;
    float spatialSamplingRadius;
  } pConst = {*param, 0, 1, frameID, 0.1f, 0.5f, 1, 32.f};

  RHI::RayTracingPassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(1280, 720, 1);
  encoder->end();
}

FinalShading::FinalShading(GIResamplingRuntimeParameters* param)
    : param(param) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "restirgi/gi-final-shading.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto FinalShading::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("GIReservoir")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Diffuse")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Specular")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  GBufferUtils::addGBufferInput(
      reflector, (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR);
  return reflector;
}

auto FinalShading::execute(RDG::RenderContext* context,
                           RDG::RenderData const& renderData) noexcept
    -> void {
  // Bind common for RT
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);
  GBufferUtils::bindGBufferResource(this, context, renderData);

  GFX::Texture* diffuse = renderData.getTexture("Diffuse");
  GFX::Texture* specular = renderData.getTexture("Specular");
  updateBinding(context, "u_Diffuse",
                RHI::BindingResource{{diffuse->getUAV(0, 0, 1)}});
  updateBinding(context, "u_Specular",
                RHI::BindingResource{{specular->getUAV(0, 0, 1)}});
  GFX::Buffer* reservoir = renderData.getBuffer("GIReservoir");
  updateBinding(context, "u_GIReservoirs",
                RHI::BindingResource{{reservoir->buffer.get(), 0, reservoir->buffer->size()}});

  const uint32_t frameID = renderData.getUInt("FrameIdx");

  struct PushConstant {
    GIResamplingRuntimeParameters params;
    uint32_t initialOutputBufferIndex;
    uint32_t enableFinalVisibility;
    uint32_t sample_batch;
  } pConst = {*param, 1, 1, frameID};

  RHI::RayTracingPassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(diffuse->texture->width(), diffuse->texture->height(), 1);
  encoder->end();
}
}