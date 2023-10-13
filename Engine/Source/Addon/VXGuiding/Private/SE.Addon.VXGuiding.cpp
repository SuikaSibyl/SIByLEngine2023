#include "../Public/SE.Addon.VXGuiding.hpp"
#include <SE.RHI.Profiler.hpp>
#include <SE.Addon.GBuffer.hpp>

namespace SIByL::Addon::VXGuiding {
VXGuiderClearPass::VXGuiderClearPass(VXGI::VXGISetting* setting,
                                     VXGuidingSetting* vxguiding_setting)
    : setting(setting), vxguiding_setting(vxguiding_setting) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "vxguiding-clear.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));

  uniformBuffer = GFX::GFXManager::get()
                      ->createStructuredUniformBuffer<VXGI::VoxelizeUniform>();
}

auto VXGuiderClearPass::reflect() noexcept -> RDG::PassReflection {
  uint32_t const size = setting->clipmapSetting.size;
  RDG::PassReflection reflector;
  reflector.addOutput("AABBMin")
      .isBuffer().withSize(size * size * size * sizeof(uint32_t) * 4)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("AABBMax")
      .isBuffer().withSize(size * size * size * sizeof(uint32_t) * 4)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("AABBMinPrebake")
      .isBuffer().withSize(size * size * size * sizeof(uint32_t) * 4)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("AABBMaxPrebake")
      .isBuffer().withSize(size * size * size * sizeof(uint32_t) * 4)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Irradiance")
      .isTexture().withSize(Math::ivec3(size, size, size))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("VPLCount")
      .isTexture().withSize(Math::ivec3(size, size, size))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("CounterBuffer")
      .isBuffer()
      .withSize(
          64)  // potential alignment requirement, could be waste of memory
      // as we only need to use a single int type here actually ...
      // But for clear meaning, we just alloc this explicitly for now.
      .withUsages((uint32_t)RHI::BufferUsage::COPY_DST |
                  (uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess((uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::TRANSFER_BIT));
  return reflector;
}

auto VXGuiderClearPass::execute(RDG::RenderContext* context,
                               RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* aabbMin = renderData.getBuffer("AABBMin");
  GFX::Buffer* aabbMax = renderData.getBuffer("AABBMax");
  GFX::Buffer* aabbMin_prebake = renderData.getBuffer("AABBMinPrebake");
  GFX::Buffer* aabbMax_prebake = renderData.getBuffer("AABBMaxPrebake");
  GFX::Texture* irradiance = renderData.getTexture("Irradiance");
  GFX::Texture* vplCount = renderData.getTexture("VPLCount");

  updateBindings(context, {
      {"u_pMin", RHI::BindingResource{{aabbMin->buffer.get(), 0, aabbMin->buffer->size()}}},
      {"u_pMax", RHI::BindingResource{{aabbMax->buffer.get(), 0, aabbMax->buffer->size()}}},
      {"u_pMinPrebake", RHI::BindingResource{{aabbMin_prebake->buffer.get(), 0, aabbMin_prebake->buffer->size()}}},
      {"u_pMaxPrebake", RHI::BindingResource{{aabbMax_prebake->buffer.get(), 0, aabbMax_prebake->buffer->size()}}},
      {"u_pIrradiance", RHI::BindingResource{{irradiance->getUAV(0, 0, 1)}}},
      {"u_vplCount", RHI::BindingResource{{vplCount->getUAV(0, 0, 1)}}},
      });

  if (vxguiding_setting->respawn) {
    RHI::ComputePassEncoder* encoder = beginPass(context);
    uint32_t const size = setting->clipmapSetting.size;
    uint32_t const dispatchSize = size / 8;
    encoder->dispatchWorkgroups(dispatchSize, dispatchSize, dispatchSize);
    encoder->end();
  }

  // Just fill buffer with 0
  int32_t zero = 0;
  const float& fzero = reinterpret_cast<const float&>(zero);
  GFX::Buffer* buffer = renderData.getBuffer("CounterBuffer");
  context->cmdEncoder->fillBuffer(buffer->buffer.get(), 0, 64, fzero);

  //
  Math::bounds3 aabb =
      *static_cast<Math::bounds3 const*>(renderData.getPtr("SceneAABB"));
  gUniform.aabbMin = aabb.pMin;
  gUniform.aabbMax = aabb.pMax;
  gUniform.voxelSize = setting->clipmapSetting.size;

  uniformBuffer.setStructure(gUniform, context->flightIdx);
  setting->shared.voxUniBinding =
      uniformBuffer.getBufferBinding(context->flightIdx);
}

auto VXGuiderClearPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Respawn", &vxguiding_setting->respawn);
  ImGui::Checkbox("Inject 1st Vertex", &vxguiding_setting->inject_first_vertex);
  ImGui::Checkbox("Use Average Irradiance", &vxguiding_setting->use_average_irradiance);
}

VXGuiderGIPass::VXGuiderGIPass(
    VXGI::VXGISetting* vx_set)
    : voxel_setting(vx_set) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vxguiding/vxguiding-gi.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto VXGuiderGIPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
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
  reflector.addInput("Irradiance")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMin")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMax")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("CounterBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("CompactIndices")
      .isBuffer()
      .withSize(131072 * sizeof(uint32_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMin")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMax")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  //reflector.addOutput("Debug")
  //    .isTexture()
  //    .withSize(Math::vec3(1, 1, 1))
  //    .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
  //    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
  //    .consume(
  //        RDG::TextureInfo::ConsumeEntry{
  //            RDG::TextureInfo::ConsumeType::StorageBinding}
  //            .addStage(
  //                (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  //reflector.addOutput("Debug1")
  //    .isTexture()
  //    .withSize(Math::vec3(1, 1, 1))
  //    .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
  //    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
  //    .consume(
  //        RDG::TextureInfo::ConsumeEntry{
  //            RDG::TextureInfo::ConsumeType::StorageBinding}
  //            .addStage(
  //                (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  //reflector.addOutput("Debug2")
  //    .isTexture()
  //    .withSize(Math::vec3(1, 1, 1))
  //    .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
  //    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
  //    .consume(
  //        RDG::TextureInfo::ConsumeEntry{
  //            RDG::TextureInfo::ConsumeType::StorageBinding}
  //            .addStage(
  //                (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("Positions")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("Node")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("TopLevelTree")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("ClusterRoots")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("SPixelIndexImage")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("FuzzyWeight")
      .isTexture().withFormat(RHI::TextureFormat::RGBA16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("FuzzyIndex")
      .isTexture().withFormat(RHI::TextureFormat::RGBA16_SINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("InverseIndex")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addInput("AssociateBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("Compact2Leaf")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("Intensity")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXGuiderGIPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* position = renderData.getTexture("Positions");
  //GFX::Texture* debug = renderData.getTexture("Debug");
  //GFX::Texture* debug1 = renderData.getTexture("Debug1");
  //GFX::Texture* debug2 = renderData.getTexture("Debug2");
  GFX::Texture* irradiance = renderData.getTexture("Irradiance");
  GFX::Texture* spxielIdx = renderData.getTexture("SPixelIndexImage");
  GFX::Texture* fweight = renderData.getTexture("FuzzyWeight");
  GFX::Texture* findex = renderData.getTexture("FuzzyIndex");
  GFX::Texture* ii = renderData.getTexture("InverseIndex");
  GFX::Texture* intensity = renderData.getTexture("Intensity");

  GFX::Buffer* counterbuffer = renderData.getBuffer("CounterBuffer");
  GFX::Buffer* compactIndices = renderData.getBuffer("CompactIndices");
  GFX::Buffer* aabbMin = renderData.getBuffer("AABBMin");
  GFX::Buffer* aabbMax = renderData.getBuffer("AABBMax");
  GFX::Buffer* treeNode = renderData.getBuffer("Node");
  GFX::Buffer* tlt = renderData.getBuffer("TopLevelTree");
  GFX::Buffer* cr = renderData.getBuffer("ClusterRoots");
  GFX::Buffer* ab = renderData.getBuffer("AssociateBuffer");
  GFX::Buffer* compact2leaf = renderData.getBuffer("Compact2Leaf");

  updateBinding(context, "u_Color",
                RHI::BindingResource{color->getUAV(0, 0, 1)});
  updateBinding(context, "u_inverseIndex",
                RHI::BindingResource{ii->getUAV(0, 0, 1)});
  updateBinding(context, "u_position",
                RHI::BindingResource{position->getUAV(0, 0, 1)});
  //updateBinding(context, "u_Debug",
  //              RHI::BindingResource{debug->getUAV(0, 0, 1)});
  //updateBinding(context, "u_Debug2",
  //              RHI::BindingResource{debug1->getUAV(0, 0, 1)});
  //updateBinding(context, "u_Debug3",
  //              RHI::BindingResource{debug2->getUAV(0, 0, 1)});
  updateBinding(context, "u_pIrradiance",
                RHI::BindingResource{{irradiance->getUAV(0, 0, 1)}});
  updateBinding(context, "u_spixelIdx",
                RHI::BindingResource{{spxielIdx->getUAV(0, 0, 1)}});
  updateBinding(context, "u_intensity",
                RHI::BindingResource{{intensity->getUAV(0, 0, 1)}});
  updateBinding(context, "u_fuzzyWeight",
                RHI::BindingResource{fweight->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_fuzzyIDX",
                RHI::BindingResource{findex->getSRV(0, 1, 0, 1)});

  updateBinding(context, "u_vxAssociate",
      RHI::BindingResource{{ab->buffer.get(), 0, ab->buffer->size()}});
  updateBinding(context, "u_topLevelTree",
      RHI::BindingResource{{tlt->buffer.get(), 0, tlt->buffer->size()}});
  updateBinding(context, "u_clusterRoots",
      RHI::BindingResource{{cr->buffer.get(), 0, cr->buffer->size()}});
  updateBinding(context, "u_pMin",
                RHI::BindingResource{
                    {aabbMin->buffer.get(), 0, aabbMin->buffer->size()}});
  updateBinding(context, "u_pMax",
                RHI::BindingResource{
                    {aabbMax->buffer.get(), 0, aabbMax->buffer->size()}});
  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{vbuffer->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_vxCounter",
                RHI::BindingResource{{counterbuffer->buffer.get(), 0,
                                      counterbuffer->buffer->size()}});
  updateBinding(context, "u_pCompactIndex",
                RHI::BindingResource{{compactIndices->buffer.get(), 0,
                                      compactIndices->buffer->size()}});
  updateBinding(context, "u_TreeNodes",
                RHI::BindingResource{{treeNode->buffer.get(), 0, treeNode->buffer->size()}});
  updateBinding(context, "u_compact2leaf",
                RHI::BindingResource{{compact2leaf->buffer.get(), 0,
                                      compact2leaf->buffer->size()}});

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);

  Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
      context->cmdEncoder, RHI::PipelineStages::TOP_OF_PIPE_BIT,
      "fused_sampling");

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("FrameIdx");

  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
    uint32_t sample_strategy;
    uint32_t second_bounce;
    uint32_t split_bar;
  };
  PushConstant pConst = {1280, 720, batchIdx,
                         (strategy & 0xf) | ((traverse_mode & 0x3) << 4) |
                             ((mis_mode & 0x1) << 6) |
                             ((visibility_mode & 0x3) << 7),
                         second ? 1 : 0,splitbar};
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(1280, 720, 1);
  encoder->end();

  Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
      context->cmdEncoder, RHI::PipelineStages::BOTTOM_OF_PIPE_BIT,
      "fused_sampling");
}

auto VXGuiderGIPass::renderUI() noexcept -> void {
  {
    const char* item_names[] = {"Init",  "BSDF",         "BSDF 2spp",
                                "Guide", "Guide + BSDF", "EXT",
                                "EXT2",  "EXT3",         "Extra Bounce"};
    ImGui::Combo("Sample Mode", &strategy, item_names, IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
  {
    const char* item_names[] = {"Unifrom", "Irradiance", "Visibility Irradiance", "SLC"};
    ImGui::Combo("VX Traverse Mode", &traverse_mode, item_names, IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
  { const char* item_names[] = {"Balance", "Power"};
    ImGui::Combo("MIS Mode", &mis_mode, item_names,
                 IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
  {
    const char* item_names[] = {"None", "Spixel", "Fuzzy Spixel"};
    ImGui::Combo("Visibility Mode", &visibility_mode, item_names,
                 IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
  ImGui::Checkbox("Second Bounce", &second);
  ImGui::DragInt("Split Bar", &splitbar, 5, 0, 1280);
  }

VXGuiderGIASVGFPass::VXGuiderGIASVGFPass(
    VXGI::VXGISetting* vx_set)
    : voxel_setting(vx_set) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vxguiding/vxguiding-gi-asvgf.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto VXGuiderGIASVGFPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("Diffuse")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
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
  reflector.addOutput("DiffuseBSDF")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("SpecularBSDF")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("RandSeed")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::R32_UINT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("RandPrev")
    .isTexture().withSize(Math::vec3(1.f))
    .withFormat(RHI::TextureFormat::R32_UINT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  GBufferUtils::addGBufferInput(
      reflector, (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR);
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
  reflector.addInput("Irradiance")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMin")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMax")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("CounterBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("CompactIndices")
      .isBuffer()
      .withSize(131072 * sizeof(uint32_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMin")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMax")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("Positions")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("Node")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("TopLevelTree")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("ClusterRoots")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("SPixelIndexImage")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("FuzzyWeight")
      .isTexture().withFormat(RHI::TextureFormat::RGBA16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("FuzzyIndex")
      .isTexture().withFormat(RHI::TextureFormat::RGBA16_SINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("InverseIndex")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addInput("AssociateBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("Compact2Leaf")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("Intensity")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXGuiderGIASVGFPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  //GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* position = renderData.getTexture("Positions");
  //GFX::Texture* debug = renderData.getTexture("Debug");
  //GFX::Texture* debug1 = renderData.getTexture("Debug1");
  //GFX::Texture* debug2 = renderData.getTexture("Debug2");
  GFX::Texture* irradiance = renderData.getTexture("Irradiance");
  GFX::Texture* spxielIdx = renderData.getTexture("SPixelIndexImage");
  GFX::Texture* fweight = renderData.getTexture("FuzzyWeight");
  GFX::Texture* findex = renderData.getTexture("FuzzyIndex");
  GFX::Texture* ii = renderData.getTexture("InverseIndex");
  GFX::Texture* intensity = renderData.getTexture("Intensity");

  GFX::Buffer* counterbuffer = renderData.getBuffer("CounterBuffer");
  GFX::Buffer* compactIndices = renderData.getBuffer("CompactIndices");
  GFX::Buffer* aabbMin = renderData.getBuffer("AABBMin");
  GFX::Buffer* aabbMax = renderData.getBuffer("AABBMax");
  GFX::Buffer* treeNode = renderData.getBuffer("Node");
  GFX::Buffer* tlt = renderData.getBuffer("TopLevelTree");
  GFX::Buffer* cr = renderData.getBuffer("ClusterRoots");
  GFX::Buffer* ab = renderData.getBuffer("AssociateBuffer");
  GFX::Buffer* compact2leaf = renderData.getBuffer("Compact2Leaf");

  GBufferUtils::bindGBufferResource(this, context, renderData);
  //GBufferUtils::bindPrevGBufferResource(this, context, renderData);

  GFX::Texture* diffuse = renderData.getTexture("Diffuse");
  GFX::Texture* specular = renderData.getTexture("Specular");
  GFX::Texture* diffuseBSDF = renderData.getTexture("DiffuseBSDF");
  GFX::Texture* specularBSDF = renderData.getTexture("SpecularBSDF");
  GFX::Texture* rand = renderData.getTexture("RandSeed");
  GFX::Texture* seedprev = renderData.getTexture("RandPrev");
  updateBinding(context, "u_Diffuse", RHI::BindingResource{{diffuse->getUAV(0, 0, 1)}});
  updateBinding(context, "u_Specular", RHI::BindingResource{{specular->getUAV(0, 0, 1)}});
  updateBinding(context, "u_DiffuseBSDF", RHI::BindingResource{{diffuseBSDF->getUAV(0, 0, 1)}});
  updateBinding(context, "u_SpecularBSDF", RHI::BindingResource{{specularBSDF->getUAV(0, 0, 1)}});
  updateBinding(context, "u_RNGSeed",
                RHI::BindingResource{{rand->getUAV(0, 0, 1)}});
  updateBinding(context, "u_RNGPrev",
                RHI::BindingResource{{seedprev->getUAV(0, 0, 1)}});


  updateBinding(context, "u_inverseIndex",
                RHI::BindingResource{ii->getUAV(0, 0, 1)});
  updateBinding(context, "u_position",
                RHI::BindingResource{position->getUAV(0, 0, 1)});
  //updateBinding(context, "u_Debug",
  //              RHI::BindingResource{debug->getUAV(0, 0, 1)});
  //updateBinding(context, "u_Debug2",
  //              RHI::BindingResource{debug1->getUAV(0, 0, 1)});
  //updateBinding(context, "u_Debug3",
  //              RHI::BindingResource{debug2->getUAV(0, 0, 1)});
  updateBinding(context, "u_pIrradiance",
                RHI::BindingResource{{irradiance->getUAV(0, 0, 1)}});
  updateBinding(context, "u_spixelIdx",
                RHI::BindingResource{{spxielIdx->getUAV(0, 0, 1)}});
  updateBinding(context, "u_intensity",
                RHI::BindingResource{{intensity->getUAV(0, 0, 1)}});
  updateBinding(context, "u_fuzzyWeight",
                RHI::BindingResource{fweight->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_fuzzyIDX",
                RHI::BindingResource{findex->getSRV(0, 1, 0, 1)});

  updateBinding(context, "u_vxAssociate",
      RHI::BindingResource{{ab->buffer.get(), 0, ab->buffer->size()}});
  updateBinding(context, "u_topLevelTree",
      RHI::BindingResource{{tlt->buffer.get(), 0, tlt->buffer->size()}});
  updateBinding(context, "u_clusterRoots",
      RHI::BindingResource{{cr->buffer.get(), 0, cr->buffer->size()}});
  updateBinding(context, "u_pMin",
                RHI::BindingResource{
                    {aabbMin->buffer.get(), 0, aabbMin->buffer->size()}});
  updateBinding(context, "u_pMax",
                RHI::BindingResource{
                    {aabbMax->buffer.get(), 0, aabbMax->buffer->size()}});
  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{vbuffer->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_vxCounter",
                RHI::BindingResource{{counterbuffer->buffer.get(), 0,
                                      counterbuffer->buffer->size()}});
  updateBinding(context, "u_pCompactIndex",
                RHI::BindingResource{{compactIndices->buffer.get(), 0,
                                      compactIndices->buffer->size()}});
  updateBinding(context, "u_TreeNodes",
                RHI::BindingResource{{treeNode->buffer.get(), 0, treeNode->buffer->size()}});
  updateBinding(context, "u_compact2leaf",
                RHI::BindingResource{{compact2leaf->buffer.get(), 0,
                                      compact2leaf->buffer->size()}});

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);

  Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
      context->cmdEncoder, RHI::PipelineStages::TOP_OF_PIPE_BIT,
      "fused_sampling");

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("FrameIdx");

  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
    uint32_t sample_strategy;
    uint32_t second_bounce;
  };
  PushConstant pConst = {1280, 720, batchIdx,
                         (strategy & 0xf) | ((traverse_mode & 0x3) << 4) |
                             ((mis_mode & 0x1) << 6) |
                             ((visibility_mode & 0x3) << 7),
                         second ? 1 : 0};
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(1280, 720, 1);
  encoder->end();

  Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
      context->cmdEncoder, RHI::PipelineStages::BOTTOM_OF_PIPE_BIT,
      "fused_sampling");
}

auto VXGuiderGIASVGFPass::renderUI() noexcept -> void {
  {
    const char* item_names[] = {"Init",  "BSDF",         "BSDF 2spp",
                                "Guide", "Guide + BSDF", "EXT",
                                "EXT2",  "EXT3",         "Extra Bounce"};
    ImGui::Combo("Sample Mode", &strategy, item_names, IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
  {
    const char* item_names[] = {"Unifrom", "Irradiance", "Visibility Irradiance", "SLC"};
    ImGui::Combo("VX Traverse Mode", &traverse_mode, item_names, IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
  { const char* item_names[] = {"Balance", "Power"};
    ImGui::Combo("MIS Mode", &mis_mode, item_names,
                 IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
  {
    const char* item_names[] = {"None", "Spixel", "Fuzzy Spixel"};
    ImGui::Combo("Visibility Mode", &visibility_mode, item_names,
                 IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
  ImGui::Checkbox("Second Bounce", &second);
  }

VXGuiderReSTIRGIPass::VXGuiderReSTIRGIPass(
    VXGI::VXGISetting* vx_set, RestirGI::GIResamplingRuntimeParameters* param)
    : voxel_setting(vx_set), param(param) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vxguiding/vxguiding-restir-initial-sample.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto VXGuiderReSTIRGIPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("VBuffer")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("Irradiance")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMin")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMax")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("CounterBuffer")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("CompactIndices")
      .isBuffer().withSize(131072 * sizeof(uint32_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMin")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMax")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("Color")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("Positions")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("Node")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("TopLevelTree")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("ClusterRoots")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("SPixelIndexImage")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("FuzzyWeight")
      .isTexture().withFormat(RHI::TextureFormat::RGBA16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("FuzzyIndex")
      .isTexture().withFormat(RHI::TextureFormat::RGBA16_SINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("InverseIndex")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addInput("AssociateBuffer")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("Compact2Leaf")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("Intensity")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("GIReservoir")
      .isBuffer().withSize(sizeof(uint32_t) * 8 * (1280 * 720 * 2))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
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

auto VXGuiderReSTIRGIPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* position = renderData.getTexture("Positions");
  GFX::Texture* irradiance = renderData.getTexture("Irradiance");
  GFX::Texture* spxielIdx = renderData.getTexture("SPixelIndexImage");
  GFX::Texture* fweight = renderData.getTexture("FuzzyWeight");
  GFX::Texture* findex = renderData.getTexture("FuzzyIndex");
  GFX::Texture* ii = renderData.getTexture("InverseIndex");
  GFX::Texture* intensity = renderData.getTexture("Intensity");

  GFX::Buffer* counterbuffer = renderData.getBuffer("CounterBuffer");
  GFX::Buffer* compactIndices = renderData.getBuffer("CompactIndices");
  GFX::Buffer* aabbMin = renderData.getBuffer("AABBMin");
  GFX::Buffer* aabbMax = renderData.getBuffer("AABBMax");
  GFX::Buffer* treeNode = renderData.getBuffer("Node");
  GFX::Buffer* tlt = renderData.getBuffer("TopLevelTree");
  GFX::Buffer* cr = renderData.getBuffer("ClusterRoots");
  GFX::Buffer* ab = renderData.getBuffer("AssociateBuffer");
  GFX::Buffer* compact2leaf = renderData.getBuffer("Compact2Leaf");

  GFX::Buffer* reservoir = renderData.getBuffer("GIReservoir");
  updateBinding(context, "u_GIReservoirs",
                RHI::BindingResource{
                    {reservoir->buffer.get(), 0, reservoir->buffer->size()}});


  updateBinding(context, "u_Color",
                RHI::BindingResource{color->getUAV(0, 0, 1)});
  updateBinding(context, "u_inverseIndex",
                RHI::BindingResource{ii->getUAV(0, 0, 1)});
  updateBinding(context, "u_position",
                RHI::BindingResource{position->getUAV(0, 0, 1)});
  updateBinding(context, "u_pIrradiance",
                RHI::BindingResource{{irradiance->getUAV(0, 0, 1)}});
  updateBinding(context, "u_spixelIdx",
                RHI::BindingResource{{spxielIdx->getUAV(0, 0, 1)}});
  updateBinding(context, "u_intensity",
                RHI::BindingResource{{intensity->getUAV(0, 0, 1)}});
  updateBinding(context, "u_fuzzyWeight",
                RHI::BindingResource{fweight->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_fuzzyIDX",
                RHI::BindingResource{findex->getSRV(0, 1, 0, 1)});

  updateBinding(context, "u_vxAssociate",
      RHI::BindingResource{{ab->buffer.get(), 0, ab->buffer->size()}});
  updateBinding(context, "u_topLevelTree",
      RHI::BindingResource{{tlt->buffer.get(), 0, tlt->buffer->size()}});
  updateBinding(context, "u_clusterRoots",
      RHI::BindingResource{{cr->buffer.get(), 0, cr->buffer->size()}});
  updateBinding(context, "u_pMin",
                RHI::BindingResource{
                    {aabbMin->buffer.get(), 0, aabbMin->buffer->size()}});
  updateBinding(context, "u_pMax",
                RHI::BindingResource{
                    {aabbMax->buffer.get(), 0, aabbMax->buffer->size()}});
  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{vbuffer->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_vxCounter",
                RHI::BindingResource{{counterbuffer->buffer.get(), 0,
                                      counterbuffer->buffer->size()}});
  updateBinding(context, "u_pCompactIndex",
                RHI::BindingResource{{compactIndices->buffer.get(), 0,
                                      compactIndices->buffer->size()}});
  updateBinding(context, "u_TreeNodes",
                RHI::BindingResource{{treeNode->buffer.get(), 0, treeNode->buffer->size()}});
  updateBinding(context, "u_compact2leaf",
                RHI::BindingResource{{compact2leaf->buffer.get(), 0,
                                      compact2leaf->buffer->size()}});

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);

  Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
      context->cmdEncoder, RHI::PipelineStages::TOP_OF_PIPE_BIT,
      "fused_sampling");

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("FrameIdx");

  struct PushConstant {
    RestirGI::GIResamplingRuntimeParameters params;
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
    uint32_t sample_strategy;
    uint32_t initialOutputBufferIndex;
  };
  PushConstant pConst = {*param, 1280, 720, batchIdx,
                         (strategy & 0xf) | ((traverse_mode & 0x3) << 4) |
                             ((mis_mode & 0x1) << 6) |
                             ((visibility_mode & 0x3) << 7)};
  pConst.initialOutputBufferIndex = 0;
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(1280, 720, 1);
  encoder->end();

  Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
      context->cmdEncoder, RHI::PipelineStages::BOTTOM_OF_PIPE_BIT,
      "fused_sampling");

  const uint32_t frameID = renderData.getUInt("FrameIdx");
  // update params
  {
    param->uniformRandomNumber = JenkinsHash(frameID);
    param->neighborOffsetMask = 8192 - 1;
  }
}

auto VXGuiderReSTIRGIPass::renderUI() noexcept -> void {
  {
    const char* item_names[] = {"Init",  "BSDF",         "BSDF 2spp",
                                "Guide", "Guide + BSDF", "EXT",
                                "EXT2",  "EXT3",         "Extra Bounce"};
    ImGui::Combo("Sample Mode", &strategy, item_names, IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
  {
    const char* item_names[] = {"Unifrom", "Irradiance", "Visibility Irradiance", "SLC"};
    ImGui::Combo("VX Traverse Mode", &traverse_mode, item_names, IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
  { const char* item_names[] = {"Balance", "Power"};
    ImGui::Combo("MIS Mode", &mis_mode, item_names,
                 IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
  {
    const char* item_names[] = {"None", "Spixel", "Fuzzy Spixel"};
    ImGui::Combo("Visibility Mode", &visibility_mode, item_names,
                 IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
}

VXGuiderAccumPass::VXGuiderAccumPass(
    VXGI::VXGISetting* vx_set)
    : voxel_setting(vx_set) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vxguiding/vxguiding-accumulator.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto VXGuiderAccumPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
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
  reflector.addInput("Irradiance")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMin")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMax")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("CounterBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("CompactIndices")
      .isBuffer()
      .withSize(131072 * sizeof(uint32_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMin")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMax")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Debug")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Debug1")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Debug2")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("Positions")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("Node")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("TopLevelTree")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("ClusterRoots")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("SPixelIndexImage")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("FuzzyWeight")
      .isTexture().withFormat(RHI::TextureFormat::RGBA16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("FuzzyIndex")
      .isTexture().withFormat(RHI::TextureFormat::RGBA16_SINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("InverseIndex")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addInput("AssociateBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("Compact2Leaf")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("PdfAccumulator")
      .isTexture().withSize(Math::ivec3(512,512,1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("PdfAccumulatorInfo")
      .isTexture().withSize(Math::ivec3(2,2,1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto VXGuiderAccumPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* pdf_accum = renderData.getTexture("PdfAccumulator");
  GFX::Texture* pdf_ainfo = renderData.getTexture("PdfAccumulatorInfo");
  updateBindings(context, {
      {"u_PdfAccumulator", RHI::BindingResource{pdf_accum->getUAV(0, 0, 1)}},
      {"u_PdfAccumulatorInfo", RHI::BindingResource{pdf_ainfo->getUAV(0, 0, 1)}}});

  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* position = renderData.getTexture("Positions");
  GFX::Texture* debug = renderData.getTexture("Debug");
  GFX::Texture* debug1 = renderData.getTexture("Debug1");
  GFX::Texture* debug2 = renderData.getTexture("Debug2");
  GFX::Texture* irradiance = renderData.getTexture("Irradiance");
  GFX::Texture* spxielIdx = renderData.getTexture("SPixelIndexImage");
  GFX::Texture* fweight = renderData.getTexture("FuzzyWeight");
  GFX::Texture* findex = renderData.getTexture("FuzzyIndex");
  GFX::Texture* ii = renderData.getTexture("InverseIndex");

  GFX::Buffer* counterbuffer = renderData.getBuffer("CounterBuffer");
  GFX::Buffer* compactIndices = renderData.getBuffer("CompactIndices");
  GFX::Buffer* aabbMin = renderData.getBuffer("AABBMin");
  GFX::Buffer* aabbMax = renderData.getBuffer("AABBMax");
  GFX::Buffer* treeNode = renderData.getBuffer("Node");
  GFX::Buffer* tlt = renderData.getBuffer("TopLevelTree");
  GFX::Buffer* cr = renderData.getBuffer("ClusterRoots");
  GFX::Buffer* ab = renderData.getBuffer("AssociateBuffer");
  GFX::Buffer* compact2leaf = renderData.getBuffer("Compact2Leaf");

  updateBinding(context, "u_Color",
                RHI::BindingResource{color->getUAV(0, 0, 1)});
  updateBinding(context, "u_inverseIndex",
                RHI::BindingResource{ii->getUAV(0, 0, 1)});
  updateBinding(context, "u_position",
                RHI::BindingResource{position->getUAV(0, 0, 1)});
  updateBinding(context, "u_Debug",
                RHI::BindingResource{debug->getUAV(0, 0, 1)});
  updateBinding(context, "u_Debug2",
                RHI::BindingResource{debug1->getUAV(0, 0, 1)});
  updateBinding(context, "u_Debug3",
                RHI::BindingResource{debug2->getUAV(0, 0, 1)});
  updateBinding(context, "u_pIrradiance",
                RHI::BindingResource{{irradiance->getUAV(0, 0, 1)}});
  updateBinding(context, "u_spixelIdx",
                RHI::BindingResource{{spxielIdx->getUAV(0, 0, 1)}});
  updateBinding(context, "u_fuzzyWeight",
                RHI::BindingResource{fweight->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_fuzzyIDX",
                RHI::BindingResource{findex->getSRV(0, 1, 0, 1)});

  updateBinding(context, "u_vxAssociate",
      RHI::BindingResource{{ab->buffer.get(), 0, ab->buffer->size()}});
  updateBinding(context, "u_topLevelTree",
      RHI::BindingResource{{tlt->buffer.get(), 0, tlt->buffer->size()}});
  updateBinding(context, "u_clusterRoots",
      RHI::BindingResource{{cr->buffer.get(), 0, cr->buffer->size()}});
  updateBinding(context, "u_pMin",
                RHI::BindingResource{
                    {aabbMin->buffer.get(), 0, aabbMin->buffer->size()}});
  updateBinding(context, "u_pMax",
                RHI::BindingResource{
                    {aabbMax->buffer.get(), 0, aabbMax->buffer->size()}});
  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{vbuffer->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_vxCounter",
                RHI::BindingResource{{counterbuffer->buffer.get(), 0,
                                      counterbuffer->buffer->size()}});
  updateBinding(context, "u_pCompactIndex",
                RHI::BindingResource{{compactIndices->buffer.get(), 0,
                                      compactIndices->buffer->size()}});
  updateBinding(context, "u_TreeNodes",
                RHI::BindingResource{{treeNode->buffer.get(), 0, treeNode->buffer->size()}});
  updateBinding(context, "u_compact2leaf",
                RHI::BindingResource{{compact2leaf->buffer.get(), 0,
                                      compact2leaf->buffer->size()}});

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("FrameIdx");

  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
    uint32_t sample_strategy;
    Math::ivec2 debug_pixel;
  };
  PushConstant pConst = {512, 512, batchIdx,
                         (strategy & 0xf) | ((traverse_mode & 0x3) << 4) |
                             ((mis_mode & 0x1) << 6) |
                             ((visibility_mode & 0x3) << 7),
                         debugPixel};

  Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
      context->cmdEncoder, RHI::PipelineStages::TOP_OF_PIPE_BIT, "sample_pass");

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(1280, 720, 1);
  encoder->end();

  Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
      context->cmdEncoder, RHI::PipelineStages::BOTTOM_OF_PIPE_BIT, "sample_pass");
}

auto VXGuiderAccumPass::onInteraction(
    Platform::Input* input, Editor::Widget::WidgetInfo* info) noexcept -> void {
  if (info->isFocused && info->isHovered) {
    if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
      debugPixel.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
      debugPixel.y = std::clamp(info->mousePos.y, 0.f, 719.f);
    }
  }
}
auto VXGuiderAccumPass::renderUI() noexcept -> void {
  {
    const char* item_names[] = {"Init",  "BSDF",         "BSDF 2spp",
                                "Guide", "Guide + BSDF", "EXT",
                                "EXT2",  "EXT3",         "Extra Bounce"};
    ImGui::Combo("Sample Mode", &strategy, item_names, IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
  {
    const char* item_names[] = {"Unifrom", "Irradiance", "Visibility Irradiance", "SLC"};
    ImGui::Combo("VX Traverse Mode", &traverse_mode, item_names, IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
  { const char* item_names[] = {"Balance", "Power"};
    ImGui::Combo("MIS Mode", &mis_mode, item_names,
                 IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
  {
    const char* item_names[] = {"None", "Spixel", "Fuzzy Spixel"};
    ImGui::Combo("Visibility Mode", &visibility_mode, item_names,
                 IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
  }
  ImGui::DragInt("Debug Pixel X", &debugPixel.x);
  ImGui::DragInt("Debug Pixel Y", &debugPixel.y);
}

VXGuiderCompactPass::VXGuiderCompactPass(VXGI::VXGISetting* setting)
    : setting(setting) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "vxguiding-compact.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VXGuiderCompactPass::reflect() noexcept -> RDG::PassReflection {
  uint32_t const size = setting->clipmapSetting.size;
  RDG::PassReflection reflector;
  reflector.addInput("Irradiance")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("InverseIndex")
      .isTexture()
      .withSize(Math::ivec3(size, size, size))
      .withFormat(RHI::TextureFormat::R32_SINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("CompactIndices")
      .isBuffer()
      .withSize(131072 * sizeof(uint32_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("CounterBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXGuiderCompactPass::execute(RDG::RenderContext* context,
                               RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* irradiance = renderData.getTexture("Irradiance");
  GFX::Texture* inverseIdx = renderData.getTexture("InverseIndex");
  GFX::Buffer* counterbuffer = renderData.getBuffer("CounterBuffer");
  GFX::Buffer* compactIndices = renderData.getBuffer("CompactIndices");

  updateBinding(context, "u_pIrradiance",
                RHI::BindingResource{{irradiance->getUAV(0, 0, 1)}});
  updateBinding(context, "u_pInverseIndex",
                RHI::BindingResource{{inverseIdx->getUAV(0, 0, 1)}});
  updateBinding(context, "u_vplCounter",
                RHI::BindingResource{{counterbuffer->buffer.get(), 0,
                                      counterbuffer->buffer->size()}});
  updateBinding(context, "u_pCompactIndex",
                RHI::BindingResource{{compactIndices->buffer.get(), 0,
                                      compactIndices->buffer->size()}});

  RHI::ComputePassEncoder* encoder = beginPass(context);
  uint32_t const size = setting->clipmapSetting.size;
  uint32_t const dispatchSize = size / 8;
  encoder->dispatchWorkgroups(dispatchSize, dispatchSize, dispatchSize);
  encoder->end();
}

VXGuiderGeometryPass::VXGuiderGeometryPass(VXGI::VXGISetting* setting)
    : setting(setting) {
  auto [vert, geom, frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "vxguiding-geome-injection.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 3>{
          std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
          std::make_pair("geometryMain", RHI::ShaderStages::GEOMETRY),
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RDG::RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(geom),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag),
      [](RHI::RenderPipelineDescriptor& desc) {
        desc.rasterize.mode =  // Enable overestimate conservative rasterization
            RHI::RasterizeState::ConservativeMode::OVERESTIMATE;
      });

  settingBuffer =
      GFX::GFXManager::get()->createStructuredUniformBuffer<VoxelSetting>();
}

auto VXGuiderGeometryPass::reflect() noexcept -> RDG::PassReflection {
  uint32_t const size = setting->clipmapSetting.size;

  RDG::PassReflection reflector;
  reflector.addInputOutput("AABBMin")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addInputOutput("AABBMax")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addInput("Irradiance")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addOutput("Depth")
      .isTexture().withSize(Math::ivec3(size, size, 1))
      .withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::DepthStencilAttachment}
                   .enableDepthWrite(false)
                   .setAttachmentLoc(0)
                   .setDepthCompareFn(RHI::CompareFunction::EQUAL));
  return reflector;
}

auto VXGuiderGeometryPass::execute(RDG::RenderContext* context,
                           RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* depth = renderData.getTexture("Depth");
  GFX::Buffer* aabbMin = renderData.getBuffer("AABBMin");
  GFX::Buffer* aabbMax = renderData.getBuffer("AABBMax");
  GFX::Texture* irradiance = renderData.getTexture("Irradiance");
  updateBinding(context, "u_pIrradiance",
                RHI::BindingResource{{irradiance->getUAV(0, 0, 1)}});

  updateBinding(context, "u_pMin",
                RHI::BindingResource{
                    {aabbMin->buffer.get(), 0, aabbMin->buffer->size()}});
  updateBinding(context, "u_pMax",
                RHI::BindingResource{
                    {aabbMax->buffer.get(), 0, aabbMax->buffer->size()}});

  renderPassDescriptor = {
      {},
      RHI::RenderPassDepthStencilAttachment{
          depth->getDSV(0, 0, 1), 1, RHI::LoadOp::DONT_CARE,
          RHI::StoreOp::DONT_CARE, false, 0, RHI::LoadOp::DONT_CARE,
          RHI::StoreOp::DONT_CARE, false},
  };

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  
  updateBinding(context, "VoxelizerUniform", setting->shared.voxUniBinding);

  settingBuffer.setStructure(gVoxelSetting, context->flightIdx);
  updateBinding(context, "VoxelConfig",
                settingBuffer.getBufferBinding(context->flightIdx));

  Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
      context->cmdEncoder, RHI::PipelineStages::TOP_OF_PIPE_BIT, "geometry_injection");

  RHI::RenderPassEncoder* encoder = beginPass(context, depth);
  if (injectGeometries) {
    renderData.getDelegate("IssueAllDrawcalls")(
        prepareDelegateData(context, renderData));
  }

  Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
      context->cmdEncoder, RHI::PipelineStages::BOTTOM_OF_PIPE_BIT, "geometry_injection");

  encoder->end();
}

auto VXGuiderGeometryPass::renderUI() noexcept -> void {
  bool use_compact = gVoxelSetting.use_compact;
  bool z_conservative = gVoxelSetting.z_conservative;
  bool clipping = gVoxelSetting.clipping;
  ImGui::Checkbox("Use Compact", &use_compact);
  ImGui::Checkbox("Z Conservative", &z_conservative);
  ImGui::Checkbox("Use Clipping", &clipping);
  ImGui::Checkbox("Inject Voxels", &injectGeometries);
  gVoxelSetting.use_compact = use_compact ? 1 : 0;
  gVoxelSetting.z_conservative = z_conservative ? 1 : 0;
  gVoxelSetting.clipping = clipping ? 1 : 0;
}

VXGuiderGeometryBakePass::VXGuiderGeometryBakePass(VXGI::VXGISetting* setting)
    : setting(setting) {
  auto [vert, geom, frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "vxguiding-geome-prebake.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 3>{
          std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
          std::make_pair("geometryMain", RHI::ShaderStages::GEOMETRY),
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RDG::RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(geom),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag),
      [](RHI::RenderPipelineDescriptor& desc) {
        desc.rasterize.mode =  // Enable overestimate conservative rasterization
            RHI::RasterizeState::ConservativeMode::OVERESTIMATE;
      });

  settingBuffer = GFX::GFXManager::get()->createStructuredUniformBuffer<VoxelSetting>();
  uniformBuffer = GFX::GFXManager::get()->createStructuredUniformBuffer<VXGI::VoxelizeUniform>();
}

auto VXGuiderGeometryBakePass::reflect() noexcept -> RDG::PassReflection {
  uint32_t const size = setting->clipmapSetting.size;
  RDG::PassReflection reflector;
  reflector.addInputOutput("AABBMin")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addInputOutput("AABBMax")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addOutput("Depth")
      .isTexture().withSize(Math::ivec3(size, size, 1))
      .withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::DepthStencilAttachment}
                   .enableDepthWrite(false)
                   .setAttachmentLoc(0)
                   .setDepthCompareFn(RHI::CompareFunction::EQUAL));
  return reflector;
}

auto VXGuiderGeometryBakePass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* depth = renderData.getTexture("Depth");
  GFX::Buffer* aabbMin = renderData.getBuffer("AABBMin");
  GFX::Buffer* aabbMax = renderData.getBuffer("AABBMax");
  updateBinding(context, "u_pMin", RHI::BindingResource{{aabbMin->buffer.get(), 0, aabbMin->buffer->size()}});
  updateBinding(context, "u_pMax", RHI::BindingResource{{aabbMax->buffer.get(), 0, aabbMax->buffer->size()}});

  renderPassDescriptor = {
      {}, RHI::RenderPassDepthStencilAttachment{
          depth->getDSV(0, 0, 1), 1, RHI::LoadOp::DONT_CARE,
          RHI::StoreOp::DONT_CARE, false, 0, RHI::LoadOp::DONT_CARE,
          RHI::StoreOp::DONT_CARE, false},
  };

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  
  Math::bounds3 aabb = *static_cast<Math::bounds3 const*>(renderData.getPtr("SceneAABB"));
  VXGI::VoxelizeUniform gUniform;
  gUniform.aabbMin = aabb.pMin;
  gUniform.aabbMax = aabb.pMax;
  gUniform.voxelSize = 64;
  uniformBuffer.setStructure(gUniform, context->flightIdx);
  updateBinding(context, "VoxelizerUniform",
                uniformBuffer.getBufferBinding(context->flightIdx));

  settingBuffer.setStructure(gVoxelSetting, context->flightIdx);
  updateBinding(context, "VoxelConfig",
                settingBuffer.getBufferBinding(context->flightIdx));

  RHI::RenderPassEncoder* encoder = beginPass(context, depth);
  renderData.getDelegate("IssueStaticDrawcalls")(
      prepareDelegateData(context, renderData));

  encoder->end();
}

VXGuiderBakeCleanPass::VXGuiderBakeCleanPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "vxguiding-bake-clear.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VXGuiderBakeCleanPass::reflect() noexcept -> RDG::PassReflection {
  uint32_t const size = 64;
  RDG::PassReflection reflector;
  reflector.addOutput("AABBMin")
      .isBuffer().withSize(size * size * size * sizeof(uint32_t) * 4)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("AABBMax")
      .isBuffer().withSize(size * size * size * sizeof(uint32_t) * 4)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXGuiderBakeCleanPass::execute(RDG::RenderContext* context,
                                RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* aabbMin = renderData.getBuffer("AABBMin");
  GFX::Buffer* aabbMax = renderData.getBuffer("AABBMax");

  updateBinding(context, "u_pMin", RHI::BindingResource{{aabbMin->buffer.get(), 0, aabbMin->buffer->size()}});
  updateBinding(context, "u_pMax", RHI::BindingResource{{aabbMax->buffer.get(), 0, aabbMax->buffer->size()}});
  
  RHI::ComputePassEncoder* encoder = beginPass(context);
  uint32_t const size = 64;
  uint32_t const dispatchSize = size / 8;
  encoder->dispatchWorkgroups(dispatchSize, dispatchSize, dispatchSize);
  encoder->end();
}

auto VXGuiderGeometryBakePass::renderUI() noexcept -> void {
  bool use_compact = gVoxelSetting.use_compact;
  bool z_conservative = gVoxelSetting.z_conservative;
  bool clipping = gVoxelSetting.clipping;
  ImGui::Checkbox("Use Compact", &use_compact);
  ImGui::Checkbox("Z Conservative", &z_conservative);
  ImGui::Checkbox("Use Clipping", &clipping);
  gVoxelSetting.use_compact = use_compact ? 1 : 0;
  gVoxelSetting.z_conservative = z_conservative ? 1 : 0;
  gVoxelSetting.clipping = clipping ? 1 : 0;
}

auto PrebakeDummyPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("AABBMin")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addOutput("AABBMax")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  return reflector;
}

auto PrebakeDummyPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Rebake", &need_rebake);
}

VXGuiderDIInjection::VXGuiderDIInjection(
    DITestSetting* test_set,
    VXGI::VXGISetting* setting)
    : di_test_setting(test_set), voxel_setting(setting) {
  auto [vert, geom, frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "testdi-injection.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 3>{
          std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
          std::make_pair("geometryMain", RHI::ShaderStages::GEOMETRY),
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RDG::RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(geom),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag),
      [](RHI::RenderPipelineDescriptor& desc) {
        desc.rasterize.mode =  // Enable overestimate conservative rasterization
            RHI::RasterizeState::ConservativeMode::OVERESTIMATE;
      });

  uniformBuffer = GFX::GFXManager::get()
                      ->createStructuredUniformBuffer<VXGI::VoxelizeUniform>();
  testdiBuffer =
      GFX::GFXManager::get()->createStructuredUniformBuffer<TestDIUniform>();
  settingBuffer =
      GFX::GFXManager::get()->createStructuredUniformBuffer<VoxelSetting>();
}

auto VXGuiderDIInjection::reflect() noexcept -> RDG::PassReflection {
  uint32_t const size = voxel_setting->clipmapSetting.size;
  RDG::PassReflection reflector;
  reflector.addInputOutput("AABBMin")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addInputOutput("AABBMax")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addInputOutput("Irradiance")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addOutput("Depth")
      .isTexture()
      .withSize(Math::ivec3(size, size, 1))
      .withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::DepthStencilAttachment}
                   .enableDepthWrite(false)
                   .setAttachmentLoc(0)
                   .setDepthCompareFn(RHI::CompareFunction::EQUAL));
  return reflector;
}

auto VXGuiderDIInjection::execute(RDG::RenderContext* context,
                                   RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* depth = renderData.getTexture("Depth");
  GFX::Buffer* aabbMin = renderData.getBuffer("AABBMin");
  GFX::Buffer* aabbMax = renderData.getBuffer("AABBMax");
  GFX::Buffer* irradiance = renderData.getBuffer("Irradiance");

  updateBinding(context, "u_pMin",
                RHI::BindingResource{
                    {aabbMin->buffer.get(), 0, aabbMin->buffer->size()}});
  updateBinding(context, "u_pMax",
                RHI::BindingResource{
                    {aabbMax->buffer.get(), 0, aabbMax->buffer->size()}});
  updateBinding(context, "u_pIrradiance",
                RHI::BindingResource{
                    {irradiance->buffer.get(), 0, irradiance->buffer->size()}});

  renderPassDescriptor = {
      {},
      RHI::RenderPassDepthStencilAttachment{
          depth->getDSV(0, 0, 1), 1, RHI::LoadOp::DONT_CARE,
          RHI::StoreOp::DONT_CARE, false, 0, RHI::LoadOp::DONT_CARE,
          RHI::StoreOp::DONT_CARE, false},
  };

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);

  Math::bounds3 aabb =
      *static_cast<Math::bounds3 const*>(renderData.getPtr("SceneAABB"));

  settingBuffer.setStructure(gVoxelSetting, context->flightIdx);
  updateBinding(context, "VoxelConfig",
                settingBuffer.getBufferBinding(context->flightIdx));

  gTestDI.worldMat =
    Math::translate({di_test_setting->position.x, di_test_setting->position.y,
                     di_test_setting->position.z})
        .m *
    Math::mat4::rotateZ(di_test_setting->rotation.z) *
    Math::mat4::rotateY(di_test_setting->rotation.y) *
    Math::mat4::rotateX(di_test_setting->rotation.x);

  gTestDI.worldMat = Math::transpose(gTestDI.worldMat);
  gTestDI.localMat = Math::inverse(gTestDI.worldMat);
  gTestDI.pMin = Math::vec2{-di_test_setting->ex, -di_test_setting->ey};
  gTestDI.pMax = Math::vec2{+di_test_setting->ex, +di_test_setting->ey};
  gTestDI.intensity = di_test_setting->intensity;
  di_test_setting->shared.DITestBinding =
      testdiBuffer.getBufferBinding(context->flightIdx);
  testdiBuffer.setStructure(gTestDI, context->flightIdx);
  updateBinding(context, "TestDISettingUniform",
                testdiBuffer.getBufferBinding(context->flightIdx));

  gUniform.aabbMin = aabb.pMin;
  gUniform.aabbMax = aabb.pMax;
  gUniform.voxelSize = voxel_setting->clipmapSetting.size;
  uniformBuffer.setStructure(gUniform, context->flightIdx);
  voxel_setting->shared.voxUniBinding =
      uniformBuffer.getBufferBinding(context->flightIdx);
  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);

  RHI::RenderPassEncoder* encoder = beginPass(context, depth);
  encoder->draw(6, 1, 0, 0);
  encoder->end();
}

auto VXGuiderDIInjection::renderUI() noexcept -> void {
  bool use_compact = gVoxelSetting.use_compact;
  bool z_conservative = gVoxelSetting.z_conservative;
  bool clipping = gVoxelSetting.clipping;
  ImGui::Checkbox("Use Compact", &use_compact);
  ImGui::Checkbox("Z Conservative", &z_conservative);
  ImGui::Checkbox("Use Clipping", &clipping);
  gVoxelSetting.use_compact = use_compact ? 1 : 0;
  gVoxelSetting.z_conservative = z_conservative ? 1 : 0;
  gVoxelSetting.clipping = clipping ? 1 : 0;
  //
  ImGui::DragFloat("ex", &di_test_setting->ex, 0.1f, 0.f, 5.f);
  ImGui::DragFloat("ey", &di_test_setting->ey, 0.1f, 0.f, 5.f);
  ImGui::DragFloat("pos.x", &di_test_setting->position.x, 0.1f);
  ImGui::DragFloat("pos.y", &di_test_setting->position.y, 0.1f);
  ImGui::DragFloat("pos.z", &di_test_setting->position.z, 0.1f);
  ImGui::DragFloat("rot.x", &di_test_setting->rotation.x, 0.1f);
  ImGui::DragFloat("rot.y", &di_test_setting->rotation.y, 0.1f);
  ImGui::DragFloat("rot.z", &di_test_setting->rotation.z, 0.1f);
  ImGui::DragFloat("intensity", &di_test_setting->intensity, 1.0f);
}

VXGuiderVisualizePass::VXGuiderVisualizePass(VXGI::VXGISetting* setting)
    : setting(setting) {
  auto [vert, frag] = GFX::ShaderLoader_SLANG::load<2u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vxguiding/vxguiding-visualize.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 2>{
          std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto VXGuiderVisualizePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("AABBMin")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT));
  reflector.addInput("AABBMax")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT));
  reflector.addInputOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA8_UNORM)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(0));
  reflector.addInputOutput("Depth")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::DepthStencilAttachment}
                   .enableDepthWrite(true)
                   .setAttachmentLoc(0)
                   .setDepthCompareFn(RHI::CompareFunction::LESS));
  reflector.addInput("Irradiance")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT));
  return reflector;
}

auto VXGuiderVisualizePass::execute(RDG::RenderContext* context,
                               RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* depth = renderData.getTexture("Depth");
  GFX::Buffer* aabbMin = renderData.getBuffer("AABBMin");
  GFX::Buffer* aabbMax = renderData.getBuffer("AABBMax");
  GFX::Texture* irradiance = renderData.getTexture("Irradiance");

  updateBinding(context, "u_pMin",
                RHI::BindingResource{
                    {aabbMin->buffer.get(), 0, aabbMin->buffer->size()}});
  updateBinding(context, "u_pMax",
                RHI::BindingResource{
                    {aabbMax->buffer.get(), 0, aabbMax->buffer->size()}});
  updateBinding(context, "u_irradiance",
                RHI::BindingResource{irradiance->getUAV(0, 0, 1)});
  updateBinding(context, "VoxelizerUniform", setting->shared.voxUniBinding);

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{color->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::LOAD,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{
          depth->getDSV(0, 0, 1), 1, RHI::LoadOp::LOAD, RHI::StoreOp::STORE,
          false, 0, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE, false},
  };

  updateBinding(context, "CameraBuffer",
                renderData.getBindingResource("GlobalUniforms").value());

  pConst.resolution.x = color->texture->width();
  pConst.resolution.y = color->texture->height();

  RHI::RenderPassEncoder* encoder = beginPass(context, color);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::VERTEX, 0,
                         sizeof(PushConstantBuffer));

  uint32_t element_num = setting->clipmapSetting.size *
                         setting->clipmapSetting.size *
                         setting->clipmapSetting.size;
  if (visualize) encoder->draw(360, element_num, 0, 0);
  encoder->end();
}

auto VXGuiderVisualizePass::renderUI() noexcept -> void {
  ImGui::DragFloat("Line Width", &pConst.line_width, 0.01f, 0);
  ImGui::Checkbox("Visualize", &visualize);
}

VXGuider1stBounceInjection::VXGuider1stBounceInjection(
    VXGI::VXGISetting* vx_set, VXGuidingSetting* vxguiding_setting)
    : voxel_setting(vx_set), vxguiding_setting(vxguiding_setting) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vxguiding/vxguiding-camera-injection.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto VXGuider1stBounceInjection::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("VBuffer")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("Irradiance")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("VPLCount")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Color")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Positions")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("ShadingPoints")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Intensity")  // only for compatibility
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXGuider1stBounceInjection::renderUI() noexcept -> void {
}

auto VXGuider1stBounceInjection::execute(
    RDG::RenderContext* context,
                               RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* sp = renderData.getTexture("ShadingPoints");
  GFX::Texture* positions = renderData.getTexture("Positions");
  GFX::Texture* irradiance = renderData.getTexture("Irradiance");
  GFX::Texture* vplCount = renderData.getTexture("VPLCount");

  updateBinding(context, "u_vplCount",
                RHI::BindingResource{{vplCount->getUAV(0, 0, 1)}});
  updateBinding(context, "u_pIrradiance",
                RHI::BindingResource{{irradiance->getUAV(0, 0, 1)}});

  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{vbuffer->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_color",
                RHI::BindingResource{color->getUAV(0, 0, 1)});
  updateBinding(context, "u_position",
                RHI::BindingResource{positions->getUAV(0, 0, 1)});
  updateBinding(context, "u_shadingpoint",
                RHI::BindingResource{sp->getUAV(0, 0, 1)});

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("FrameIdx");

  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
    uint32_t setting;
  };
  PushConstant pConst = {1280, 720, batchIdx, 0};
  if (vxguiding_setting->inject_first_vertex) pConst.setting |= 0b1;
  if (vxguiding_setting->use_average_irradiance) pConst.setting |= 0b10;
  
  Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
      context->cmdEncoder, RHI::PipelineStages::TOP_OF_PIPE_BIT, "light_injection");

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  if (vxguiding_setting->respawn) {
    encoder->traceRays(1280, 720, 1);
  }
  encoder->end();
  
  Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
      context->cmdEncoder, RHI::PipelineStages::BOTTOM_OF_PIPE_BIT, "light_injection");
}

VXGuiderLightInjection::VXGuiderLightInjection(
    VXGI::VXGISetting* vx_set, VXGuidingSetting* vxguiding_setting)
    : voxel_setting(vx_set), vxguiding_setting(vxguiding_setting) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "vxguiding-light-injection.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VXGuiderLightInjection::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("VBuffer")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("Irradiance")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("VPLCount")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Color")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Intensity")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Positions")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("ShadingPoints")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXGuiderLightInjection::renderUI() noexcept -> void {}

auto VXGuiderLightInjection::execute(
    RDG::RenderContext* context,
                               RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* intensity = renderData.getTexture("Intensity");
  GFX::Texture* sp = renderData.getTexture("ShadingPoints");
  GFX::Texture* positions = renderData.getTexture("Positions");
  GFX::Texture* irradiance = renderData.getTexture("Irradiance");
  GFX::Texture* vplCount = renderData.getTexture("VPLCount");

  updateBindings(context, {
    {"u_vplCount", RHI::BindingResource{{vplCount->getUAV(0, 0, 1)}}},
    {"u_vBuffer", RHI::BindingResource{vbuffer->getSRV(0, 1, 0, 1)}},
    {"u_intensity", RHI::BindingResource{intensity->getUAV(0, 0, 1)}},
    {"u_position", RHI::BindingResource{positions->getUAV(0, 0, 1)}},
    {"u_shadingpoint", RHI::BindingResource{sp->getUAV(0, 0, 1)}},
    {"u_pIrradiance", RHI::BindingResource{{irradiance->getUAV(0, 0, 1)}}},
    {"VoxelizerUniform", voxel_setting->shared.voxUniBinding},
  });

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);

  uint32_t batchIdx = renderData.getUInt("FrameIdx");
  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
    uint32_t setting;
  };
  PushConstant pConst = {1280, 720, batchIdx, 0};

  RHI::ComputePassEncoder* encoder = beginPass(context);
  if (vxguiding_setting->inject_first_vertex) pConst.setting |= 0b1;
  if (vxguiding_setting->use_average_irradiance) pConst.setting |= 0b10;
  Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
      context->cmdEncoder, RHI::PipelineStages::TOP_OF_PIPE_BIT, "light_injection");

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((1280 + 15) / 16, (720 + 15) / 16, 1);
  encoder->end();
  
  Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
      context->cmdEncoder, RHI::PipelineStages::BOTTOM_OF_PIPE_BIT, "light_injection");
}

VoxelClear6DPass::VoxelClear6DPass(VXGI::VXGISetting* setting)
    : setting(setting) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxgi/"
      "voxel-6d-clear.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));

    // Create sampler
  Core::GUID samplerGUID =
      Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
  GFX::GFXManager::get()->registerSamplerResource(
      samplerGUID,
      RHI::SamplerDescriptor{
          RHI::AddressMode::CLAMP_TO_EDGE, RHI::AddressMode::CLAMP_TO_EDGE,
          RHI::AddressMode::CLAMP_TO_EDGE, RHI::FilterMode::LINEAR,
          RHI::FilterMode::LINEAR, RHI::MipmapFilterMode::NEAREST, 0.f, 32.f,
          RHI::CompareFunction::ALWAYS, 1,
          float(setting->clipmapSetting.mip)});
  setting->shared.tex3dSampler =
      Core::ResourceManager::get()->getResource<GFX::Sampler>(samplerGUID);
}

auto VoxelClear6DPass::reflect() noexcept -> RDG::PassReflection {
  uint32_t const size = setting->clipmapSetting.size;
    
  RDG::PassReflection reflector;

  for (int i = 0; i < 6; ++i)
    reflector.addOutput("RadOpaVox6DTex" + std::to_string(i))
        .isTexture()
        .withSize(Math::ivec3(size, size, size))
        .withLevels(setting->clipmapSetting.mip)
        .withFormat(RHI::TextureFormat::RG16_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::StorageBinding}
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT)
                .setSubresource(0, 1, 0, 1));

  return reflector;
}

auto VoxelClear6DPass::execute(RDG::RenderContext* context,
                             RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* radopaTex[6];
  for (int i = 0; i < 6; ++i)
    radopaTex[i] = renderData.getTexture("RadOpaVox6DTex" + std::to_string(i));

  std::vector<RHI::TextureView*> tex6DArray{
      radopaTex[0]->getUAV(0, 0, 1),
      radopaTex[1]->getUAV(0, 0, 1),
      radopaTex[2]->getUAV(0, 0, 1),
      radopaTex[3]->getUAV(0, 0, 1),
      radopaTex[4]->getUAV(0, 0, 1),
      radopaTex[5]->getUAV(0, 0, 1),
  };
  updateBinding(context, "uRadOpaVox6D", RHI::BindingResource{tex6DArray});

  RHI::ComputePassEncoder* encoder = beginPass(context);
  uint32_t const size = setting->clipmapSetting.size;
  uint32_t const dispatchSize = size / 8;
  encoder->dispatchWorkgroups(dispatchSize, dispatchSize, dispatchSize);
  encoder->end();
}

VXGuiderViewPass::VXGuiderViewPass(VXGI::VXGISetting* setting)
    : setting(setting) {
  auto [frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "vxguiding-viewer.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RDG::FullScreenPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto VXGuiderViewPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("Irradiance")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA8_UNORM)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(0));
  reflector.addInput("InverseIndex")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addInput("AssociateBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VXNormal")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("SPixelVisibility")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("SPixelIndexImage")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VXPremulIrradiance")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXGuiderViewPass::execute(RDG::RenderContext* context,
                                   RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* visibility = renderData.getTexture("SPixelVisibility");
  GFX::Texture* index = renderData.getTexture("SPixelIndexImage");
  GFX::Buffer* vxNormal = renderData.getBuffer("VXNormal");
  GFX::Texture* inverseID = renderData.getTexture("InverseIndex");
  GFX::Buffer* associate = renderData.getBuffer("AssociateBuffer");
  GFX::Buffer* premulIrradiance = renderData.getBuffer("VXPremulIrradiance");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{color->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{},
  };

  updateBindings(
      context, std::vector<std::pair<std::string, RHI::BindingResource>>{
                   std::make_pair(
                       "CameraBuffer",
                       renderData.getBindingResource("GlobalUniforms").value()),
               });
  updateBinding(context, "VoxelizerUniform", setting->shared.voxUniBinding);

  GFX::Texture* irradianceTex = renderData.getTexture("Irradiance");
  updateBinding(context, "uIrradianceMip",
                RHI::BindingResource{irradianceTex->getSRV(0, 1, 0, 1)});
  updateBinding(context, "uSPixelVisibility",
                RHI::BindingResource{visibility->getSRV(0, 1, 0, 1)});
  updateBinding(context, "uSPixelIndex",
                RHI::BindingResource{index->getSRV(0, 1, 0, 1)});
  updateBinding(context, "uInverseIndex",
                RHI::BindingResource{inverseID->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_associate",
                RHI::BindingResource{
                    {associate->buffer.get(), 0, associate->buffer->size()}});
  updateBinding(context, "u_vxNormal",
                RHI::BindingResource{
                    {vxNormal->buffer.get(), 0, vxNormal->buffer->size()}});
  updateBinding(context, "u_premulIrr",
                RHI::BindingResource{{premulIrradiance->buffer.get(), 0,
                                      premulIrradiance->buffer->size()}});

  RHI::RenderPassEncoder* encoder = beginPass(context, color);

  pConst.resolution = Math::ivec2{1280, 720};
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::FRAGMENT, 0,
                         sizeof(PushConstant));

  dispatchFullScreen(context);

  encoder->end();
}

auto VXGuiderViewPass::renderUI() noexcept -> void {
  ImGui::DragFloat("Edge Threshold", &pConst.edgeThreshold, 0.005, 0, 1);
  ImGui::DragFloat("Show scalar", &pConst.scalar, 1);
  ImGui::DragInt("Show Enum", &pConst.showWhat, 1, 0, 2);
  ImGui::DragInt("Pixel X", &pConst.debugPixel.x);
  ImGui::DragInt("Pixel Y", &pConst.debugPixel.y);
}

auto VXGuiderViewPass::onInteraction(Platform::Input* input,
    Editor::Widget::WidgetInfo* info) noexcept
    -> void {
  if (info->isFocused && info->isHovered) {
    if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
      pConst.debugPixel.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
      pConst.debugPixel.y = std::clamp(info->mousePos.y, 0.f, 719.f);
    }
  }
}

Voxelize6DPass::Voxelize6DPass(VXGI::VXGISetting* setting) : setting(setting) {
  auto [vert, geom, frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxgi/"
      "voxel-6d-voxelizer",
      std::array<std::pair<std::string, RHI::ShaderStages>, 3>{
          std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
          std::make_pair("geometryMain", RHI::ShaderStages::GEOMETRY),
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RDG::RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(geom),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag),
      [](RHI::RenderPipelineDescriptor& desc) {
        desc.rasterize.mode =  // Enable overestimate conservative rasterization
            RHI::RasterizeState::ConservativeMode::OVERESTIMATE;
      });

  uniformBuffer = GFX::GFXManager::get()
                      ->createStructuredUniformBuffer<VXGI::VoxelizeUniform>();
}

auto Voxelize6DPass::reflect() noexcept -> RDG::PassReflection {
  uint32_t const size = setting->clipmapSetting.size;

  RDG::PassReflection reflector;

  for (int i = 0; i < 6; ++i)
    reflector.addInputOutput("RadOpaVox6DTex" + std::to_string(i))
        .isTexture()
        .withSize(Math::ivec3(size, size, size))
        .withFormat(RHI::TextureFormat::RG16_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::StorageBinding}
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT)
                .setSubresource(0, 1, 0, 1));

  reflector.addOutput("Depth")
      .isTexture()
      .withSize(Math::ivec3(size, size, 1))
      .withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::DepthStencilAttachment}
                   .enableDepthWrite(false)
                   .setAttachmentLoc(0)
                   .setDepthCompareFn(RHI::CompareFunction::EQUAL));

  return reflector;
}

auto Voxelize6DPass::execute(RDG::RenderContext* context,
                           RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* depth = renderData.getTexture("Depth");

  GFX::Texture* radopaTex[6];
  for (int i = 0; i < 6; ++i)
    radopaTex[i] = renderData.getTexture("RadOpaVox6DTex" + std::to_string(i));

  renderPassDescriptor = {
      {},
      RHI::RenderPassDepthStencilAttachment{
          depth->getDSV(0, 0, 1), 1, RHI::LoadOp::DONT_CARE,
          RHI::StoreOp::DONT_CARE, false, 0, RHI::LoadOp::DONT_CARE,
          RHI::StoreOp::DONT_CARE, false},
  };

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);

  std::vector<RHI::TextureView*> tex6DArray{
      radopaTex[0]->getUAV(0, 0, 1), radopaTex[1]->getUAV(0, 0, 1),
      radopaTex[2]->getUAV(0, 0, 1), radopaTex[3]->getUAV(0, 0, 1),
      radopaTex[4]->getUAV(0, 0, 1), radopaTex[5]->getUAV(0, 0, 1),
  };
  updateBinding(context, "uRadOpaVox6D", RHI::BindingResource{tex6DArray});

  Math::bounds3 aabb =
      *static_cast<Math::bounds3 const*>(renderData.getPtr("SceneAABB"));
  gUniform.aabbMin = aabb.pMin;
  gUniform.aabbMax = aabb.pMax;
  gUniform.voxelSize = setting->clipmapSetting.size;

  uniformBuffer.setStructure(gUniform, context->flightIdx);
  setting->shared.voxUniBinding =
      uniformBuffer.getBufferBinding(context->flightIdx);
  updateBinding(context, "VoxelizerUniform", setting->shared.voxUniBinding);

  RHI::RenderPassEncoder* encoder = beginPass(context, depth);

  renderData.getDelegate("IssueAllDrawcalls")(
      prepareDelegateData(context, renderData));

  encoder->end();
}

VoxelMip6DPass::VoxelMip6DPass(VXGI::VXGISetting* setting, bool secondPass)
    : setting(setting), secondPass(secondPass) {
  std::vector<std::pair<char const*, char const*>> macros;
  if (secondPass) {    // only first pass need to build mip3
    macros.emplace_back(std::make_pair("SECOND_PASS", "1"));
  }
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxgi/"
      "voxel-6d-downsampling.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      },
      macros);

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VoxelMip6DPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  const uint32_t mip_beg = secondPass ? 3 : 0;
  const uint32_t mip_end = secondPass ? 6 : 4;

  for (int i = 0; i < 6; ++i)
    reflector.addInputOutput("RadOpaVox6DTex" + std::to_string(i))
        .isTexture()
        .withFormat(RHI::TextureFormat::RG16_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::StorageBinding}
                .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT)
                .setSubresource(mip_beg, mip_end, 0, 1));

  return reflector;
}

auto VoxelMip6DPass::execute(RDG::RenderContext* context,
                           RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* radopaTex[6];
  for (int i = 0; i < 6; ++i)
    radopaTex[i] = renderData.getTexture("RadOpaVox6DTex" + std::to_string(i));

  if (!secondPass) {
    updateBinding(context, "uRadOpaVox6D",
                  RHI::BindingResource{std::vector<RHI::TextureView*>{
                      radopaTex[0]->getUAV(0, 0, 1),
                      radopaTex[1]->getUAV(0, 0, 1),
                      radopaTex[2]->getUAV(0, 0, 1),
                      radopaTex[3]->getUAV(0, 0, 1),
                      radopaTex[4]->getUAV(0, 0, 1),
                      radopaTex[5]->getUAV(0, 0, 1),
                  }});
    updateBinding(context, "uRadOpaVox6D_MIP1",
                  RHI::BindingResource{std::vector<RHI::TextureView*>{
                      radopaTex[0]->getUAV(1, 0, 1),
                      radopaTex[1]->getUAV(1, 0, 1),
                      radopaTex[2]->getUAV(1, 0, 1),
                      radopaTex[3]->getUAV(1, 0, 1),
                      radopaTex[4]->getUAV(1, 0, 1),
                      radopaTex[5]->getUAV(1, 0, 1),
                  }});
    updateBinding(context, "uRadOpaVox6D_MIP2",
                  RHI::BindingResource{std::vector<RHI::TextureView*>{
                      radopaTex[0]->getUAV(2, 0, 1),
                      radopaTex[1]->getUAV(2, 0, 1),
                      radopaTex[2]->getUAV(2, 0, 1),
                      radopaTex[3]->getUAV(2, 0, 1),
                      radopaTex[4]->getUAV(2, 0, 1),
                      radopaTex[5]->getUAV(2, 0, 1),
                  }});
    updateBinding(context, "uRadOpaVox6D_MIP3",
                  RHI::BindingResource{std::vector<RHI::TextureView*>{
                      radopaTex[0]->getUAV(3, 0, 1),
                      radopaTex[1]->getUAV(3, 0, 1),
                      radopaTex[2]->getUAV(3, 0, 1),
                      radopaTex[3]->getUAV(3, 0, 1),
                      radopaTex[4]->getUAV(3, 0, 1),
                      radopaTex[5]->getUAV(3, 0, 1),
                  }});
  } else {
    updateBinding(context, "uRadOpaVox6D",
                  RHI::BindingResource{std::vector<RHI::TextureView*>{
                      radopaTex[0]->getUAV(3, 0, 1),
                      radopaTex[1]->getUAV(3, 0, 1),
                      radopaTex[2]->getUAV(3, 0, 1),
                      radopaTex[3]->getUAV(3, 0, 1),
                      radopaTex[4]->getUAV(3, 0, 1),
                      radopaTex[5]->getUAV(3, 0, 1),
                  }});
    updateBinding(context, "uRadOpaVox6D_MIP1",
                  RHI::BindingResource{std::vector<RHI::TextureView*>{
                      radopaTex[0]->getUAV(4, 0, 1),
                      radopaTex[1]->getUAV(4, 0, 1),
                      radopaTex[2]->getUAV(4, 0, 1),
                      radopaTex[3]->getUAV(4, 0, 1),
                      radopaTex[4]->getUAV(4, 0, 1),
                      radopaTex[5]->getUAV(4, 0, 1),
                  }});
    updateBinding(context, "uRadOpaVox6D_MIP2",
                  RHI::BindingResource{std::vector<RHI::TextureView*>{
                      radopaTex[0]->getUAV(5, 0, 1),
                      radopaTex[1]->getUAV(5, 0, 1),
                      radopaTex[2]->getUAV(5, 0, 1),
                      radopaTex[3]->getUAV(5, 0, 1),
                      radopaTex[4]->getUAV(5, 0, 1),
                      radopaTex[5]->getUAV(5, 0, 1),
                  }});
  }

  RHI::ComputePassEncoder* encoder = beginPass(context);
  uint32_t const size = setting->clipmapSetting.size;
  uint32_t const dispatchSize = secondPass ? 1 : size / 8;
  struct PushConstant {
    int saveNumericalError;
  } pConst;
  pConst.saveNumericalError = saveNumericalError ? 1 : 0;
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  if (updateMip) {
    encoder->dispatchWorkgroups(dispatchSize, dispatchSize, dispatchSize * 6);
  }
  encoder->end();
}

auto VoxelMip6DPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Update MIP", &updateMip);
  ImGui::Checkbox("Save Digital Error", &saveNumericalError);
}

Voxel6DRTInjection::Voxel6DRTInjection(VXGI::VXGISetting* vx_set)
    : voxel_setting(vx_set) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vxgi/voxel-6d-injection-rt.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto Voxel6DRTInjection::reflect() noexcept -> RDG::PassReflection {
  uint32_t const size = voxel_setting->clipmapSetting.size;

  RDG::PassReflection reflector;

  for (int i = 0; i < 6; ++i)
    reflector.addInputOutput("RadOpaVox6DTex" + std::to_string(i))
        .isTexture()
        .withSize(Math::ivec3(size, size, size))
        .withFormat(RHI::TextureFormat::RG16_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::StorageBinding}
                .addStage(
                    (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR)
                .setSubresource(0, 1, 0, 1));
  
  return reflector;
}

auto Voxel6DRTInjection::execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* radopaTex[6];
  for (int i = 0; i < 6; ++i)
    radopaTex[i] = renderData.getTexture("RadOpaVox6DTex" + std::to_string(i));

  std::vector<RHI::TextureView*> tex6DArray{
      radopaTex[0]->getUAV(0, 0, 1), radopaTex[1]->getUAV(0, 0, 1),
      radopaTex[2]->getUAV(0, 0, 1), radopaTex[3]->getUAV(0, 0, 1),
      radopaTex[4]->getUAV(0, 0, 1), radopaTex[5]->getUAV(0, 0, 1),
  };

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  // binding resources
  updateBinding(context, "uRadOpaVox6D", RHI::BindingResource{tex6DArray});
  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("FrameIdx");

  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
  };
  PushConstant pConst = {1280, 720, batchIdx};

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(1280, 720, 1);

  encoder->end();
}

Voxel6DVisualizePass::Voxel6DVisualizePass(VXGI::VXGISetting* setting,
                                           bool isotropic)
    : setting(setting), isotropic(isotropic) {
  std::vector<std::pair<char const*, char const*>> macros;
  if (isotropic) {  // only first pass need to build mip3
    macros.emplace_back(std::make_pair("USE_ISOTROPIC_VOXEL", "1"));
  }

  auto [frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxgi/"
      "voxel-6d-viewer.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      },
      macros);

  RDG::FullScreenPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

 auto Voxel6DVisualizePass::reflect() noexcept -> RDG::PassReflection {
  uint32_t const size = setting->clipmapSetting.size;

  RDG::PassReflection reflector;

  if (isotropic) {
    reflector.addInputOutput("RadOpaVox6DTex")
        .isTexture()
        .withSize(Math::ivec3(size, size, size))
        .withFormat(RHI::TextureFormat::RG16_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::StorageBinding}
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT)
                .setSubresource(0, setting->clipmapSetting.mip, 0, 1));
  } else {
    for (int i = 0; i < 6; ++i)
      reflector.addInputOutput("RadOpaVox6DTex" + std::to_string(i))
          .isTexture()
          .withSize(Math::ivec3(size, size, size))
          .withFormat(RHI::TextureFormat::RG16_FLOAT)
          .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
          .consume(
              RDG::TextureInfo::ConsumeEntry{
                  RDG::TextureInfo::ConsumeType::StorageBinding}
                  .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT)
                  .setSubresource(0, setting->clipmapSetting.mip, 0, 1));
  }

  reflector.addOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA8_UNORM)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(0));

  return reflector;
}

auto Voxel6DVisualizePass::execute(RDG::RenderContext* context,
                                 RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{color->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{},
  };

  updateBindings(
      context, std::vector<std::pair<std::string, RHI::BindingResource>>{
                   std::make_pair(
                       "CameraBuffer",
                       renderData.getBindingResource("GlobalUniforms").value()),
               });
  updateBinding(context, "VoxelizerUniform", setting->shared.voxUniBinding);

  if (isotropic) {
    GFX::Texture* radopaTex = renderData.getTexture("RadOpaVox6DTex");
    updateBinding(context, "uRadOpaVox6D",
                  RHI::BindingResource{
                      radopaTex->getSRV(0, setting->clipmapSetting.mip, 0, 1)});

  } else {
    GFX::Texture* radopaTex[6];
    for (int i = 0; i < 6; ++i)
      radopaTex[i] =
          renderData.getTexture("RadOpaVox6DTex" + std::to_string(i));
    updateBinding(
        context, "uRadOpaVox6D",
        RHI::BindingResource{std::vector<RHI::TextureView*>{
            radopaTex[0]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
            radopaTex[1]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
            radopaTex[2]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
            radopaTex[3]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
            radopaTex[4]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
            radopaTex[5]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
        }});
  }
  updateBinding(
      context, "uTex3DSampler",
      RHI::BindingResource{setting->shared.tex3dSampler->sampler.get()});


  RHI::RenderPassEncoder* encoder = beginPass(context, color);

  pConst.resolution = Math::ivec2{1280, 720};

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::FRAGMENT, 0,
                         sizeof(PushConstant));

  dispatchFullScreen(context);

  encoder->end();
}

auto Voxel6DVisualizePass::renderUI() noexcept -> void {
  ImGui::DragFloat("Edge Threshold", &pConst.edgeThreshold, 0.005, 0, 1);
  ImGui::DragFloat("Value Scalar", &pConst.scaling, 0.05, 0, 100);
  ImGui::DragInt("Mip Level", &pConst.mipLevelShown, 1, 0,
                 setting->clipmapSetting.mip - 1);
  { // Select an item type
    const char* item_names[] = {"Luminance", "Opacity", "Fully Opaque", "Lumin Only"};
    int debug_mode = int(pConst.showWhat);
    ImGui::Combo("Mode", &debug_mode, item_names, IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
    pConst.showWhat = debug_mode;
  }
  bool useSampler = pConst.useSampler != 0;
  ImGui::Checkbox("Use Sampler", &useSampler);
  pConst.useSampler = useSampler ? 1 : 0;
}

DITestInjectPass::DITestInjectPass(DITestSetting* test_set,
                                   VXGI::VXGISetting* voxel_setting)
    : di_test_setting(test_set), voxel_setting(voxel_setting) {
  auto [vert, geom, frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "testdi-injection.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 3>{
          std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
          std::make_pair("geometryMain", RHI::ShaderStages::GEOMETRY),
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RDG::RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(geom),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag),
      [](RHI::RenderPipelineDescriptor& desc) {
        desc.rasterize.mode =  // Enable overestimate conservative rasterization
            RHI::RasterizeState::ConservativeMode::OVERESTIMATE;
      });

  uniformBuffer =
      GFX::GFXManager::get()->createStructuredUniformBuffer<TestDIUniform>();
}

auto DITestInjectPass::reflect() noexcept -> RDG::PassReflection {
  uint32_t const size = voxel_setting->clipmapSetting.size;
    
  RDG::PassReflection reflector;

  for (int i = 0; i < 6; ++i)
    reflector.addInputOutput("RadOpaVox6DTex" + std::to_string(i))
        .isTexture()
        .withFormat(RHI::TextureFormat::RG16_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::StorageBinding}
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT)
                .setSubresource(0, 1, 0, 1));
  
  reflector.addInput("Depth")
      .isTexture()
      .withSize(Math::ivec3(size, size, 1))
      .withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::DepthStencilAttachment}
                   .enableDepthWrite(false)
                   .setAttachmentLoc(0)
                   .setDepthCompareFn(RHI::CompareFunction::EQUAL));
  
  return reflector;
}

auto DITestInjectPass::execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* depth = renderData.getTexture("Depth");
  GFX::Texture* radopaTex[6];
  for (int i = 0; i < 6; ++i)
    radopaTex[i] = renderData.getTexture("RadOpaVox6DTex" + std::to_string(i));

  renderPassDescriptor = {
      {},
      RHI::RenderPassDepthStencilAttachment{
          depth->getDSV(0, 0, 1), 1, RHI::LoadOp::DONT_CARE,
          RHI::StoreOp::DONT_CARE, false, 0, RHI::LoadOp::DONT_CARE,
          RHI::StoreOp::DONT_CARE, false},
  };

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);

  gUniform.worldMat =
      Math::translate({di_test_setting->position.x, di_test_setting->position.y,
                       di_test_setting->position.z})
          .m *
      Math::mat4::rotateZ(di_test_setting->rotation.z) *
      Math::mat4::rotateY(di_test_setting->rotation.y) *
      Math::mat4::rotateX(di_test_setting->rotation.x);

  gUniform.worldMat = Math::transpose(gUniform.worldMat);
  gUniform.localMat = Math::inverse(gUniform.worldMat);
  gUniform.pMin = Math::vec2{-di_test_setting->ex, -di_test_setting->ey};
  gUniform.pMax = Math::vec2{+di_test_setting->ex, +di_test_setting->ey};
  gUniform.intensity = di_test_setting->intensity;
  uniformBuffer.setStructure(gUniform, context->flightIdx);
  di_test_setting->shared.DITestBinding =
      uniformBuffer.getBufferBinding(context->flightIdx);

  std::vector<RHI::TextureView*> tex6DArray{
      radopaTex[0]->getUAV(0, 0, 1),
      radopaTex[1]->getUAV(0, 0, 1),
      radopaTex[2]->getUAV(0, 0, 1),
      radopaTex[3]->getUAV(0, 0, 1),
      radopaTex[4]->getUAV(0, 0, 1),
      radopaTex[5]->getUAV(0, 0, 1),
  };
  updateBinding(context, "uRadopaVox6D", RHI::BindingResource{tex6DArray});

  updateBinding(context, "TestDISettingUniform",
                di_test_setting->shared.DITestBinding);
  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);

  RHI::RenderPassEncoder* encoder = beginPass(context, depth);
  encoder->draw(6, 1, 0, 0);
  encoder->end();
}

auto DITestInjectPass::renderUI() noexcept -> void {
  ImGui::DragFloat("ex", &di_test_setting->ex, 0.1f, 0.f, 5.f);
  ImGui::DragFloat("ey", &di_test_setting->ey, 0.1f, 0.f, 5.f);
  ImGui::DragFloat("pos.x", &di_test_setting->position.x, 0.1f);
  ImGui::DragFloat("pos.y", &di_test_setting->position.y, 0.1f);
  ImGui::DragFloat("pos.z", &di_test_setting->position.z, 0.1f);
  ImGui::DragFloat("rot.x", &di_test_setting->rotation.x, 0.1f);
  ImGui::DragFloat("rot.y", &di_test_setting->rotation.y, 0.1f);
  ImGui::DragFloat("rot.z", &di_test_setting->rotation.z, 0.1f);
  ImGui::DragFloat("intensity", &di_test_setting->intensity, 1.0f);
}

DITestVoxelCheckPass::DITestVoxelCheckPass(DITestSetting* test_set,
                                           VXGI::VXGISetting* setting)
    : setting(setting), di_test_setting(test_set) {
  auto [frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "testdi-voxelize-check.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RDG::FullScreenPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto DITestVoxelCheckPass::reflect() noexcept -> RDG::PassReflection {
  uint32_t const size = setting->clipmapSetting.size;

  RDG::PassReflection reflector;

  for (int i = 0; i < 6; ++i)
    reflector.addInput("RadOpaVox6DTex" + std::to_string(i))
        .isTexture()
        .withFormat(RHI::TextureFormat::RG16_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::StorageBinding}
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT)
                .setSubresource(0, setting->clipmapSetting.mip, 0, 1));

  reflector.addOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA8_UNORM)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(0));

  return reflector;
}

auto DITestVoxelCheckPass::execute(RDG::RenderContext* context,
                                   RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* radopaTex[6];
  for (int i = 0; i < 6; ++i)
    radopaTex[i] = renderData.getTexture("RadOpaVox6DTex" + std::to_string(i));

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{color->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{},
  };

  updateBindings(
      context, std::vector<std::pair<std::string, RHI::BindingResource>>{
                   std::make_pair(
                       "CameraBuffer",
                       renderData.getBindingResource("GlobalUniforms").value()),
               });
  updateBinding(context, "VoxelizerUniform", setting->shared.voxUniBinding);
  std::vector<RHI::TextureView*> tex6DArray{
      radopaTex[0]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
      radopaTex[1]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
      radopaTex[2]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
      radopaTex[3]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
      radopaTex[4]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
      radopaTex[5]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
  };
  updateBinding(context, "uRadopaVox6D", RHI::BindingResource{tex6DArray});

  updateBinding(context, "TestDISettingUniform",
                di_test_setting->shared.DITestBinding);

  RHI::RenderPassEncoder* encoder = beginPass(context, color);

  pConst.resolution = Math::ivec2{1280, 720};

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::FRAGMENT, 0,
                         sizeof(PushConstant));

  dispatchFullScreen(context);

  encoder->end();
}

auto DITestVoxelCheckPass::renderUI() noexcept -> void {
  ImGui::DragFloat("Edge Threshold", &pConst.edgeThreshold, 0.005, 0, 1);
  ImGui::DragFloat("Value Scalar", &pConst.scaling, 0.05, 0, 100);
  ImGui::DragInt("Mip Level", &pConst.mipLevelShown, 1, 0,
                 setting->clipmapSetting.mip - 1);
  {  // Select an item type
    const char* item_names[] = {"Luminance", "Opacity"};
    int debug_mode = int(pConst.showWhat);
    ImGui::Combo("Mode", &debug_mode, item_names, IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
    pConst.showWhat = debug_mode;
  }
}

DITestPass::DITestPass(DITestSetting* test_set, VXGI::VXGISetting* vx_set)
    : di_test_setting(test_set), voxel_setting(vx_set) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vxguiding/testdi-compare.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto DITestPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

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
  reflector.addInput("CompactIndices")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("CounterBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("DebugTex")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMin")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("AABBMax")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

  return reflector;
}

auto DITestPass::renderUI() noexcept -> void {
  ImGui::DragInt("Strategy", &strategy);
  //ImGui::DragFloat("Cone Length", &coneLength);
  //{
  //  const char* item_names[] = {"DI Only", "GI Only", "Combined"};
  //  ImGui::Combo("Mode", &showMode, item_names, IM_ARRAYSIZE(item_names),
  //               IM_ARRAYSIZE(item_names));
  //}
}

auto DITestPass::execute(RDG::RenderContext* context,
                                    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* debug = renderData.getTexture("DebugTex");
  GFX::Buffer* compactIndices = renderData.getBuffer("CompactIndices");
  GFX::Buffer* counterbuffer = renderData.getBuffer("CounterBuffer");
  GFX::Buffer* aabbMin = renderData.getBuffer("AABBMin");
  GFX::Buffer* aabbMax = renderData.getBuffer("AABBMax");

  updateBinding(context, "u_pMin",
                RHI::BindingResource{
                    {aabbMin->buffer.get(), 0, aabbMin->buffer->size()}});
  updateBinding(context, "u_pMax",
                RHI::BindingResource{
                    {aabbMax->buffer.get(), 0, aabbMax->buffer->size()}});

  updateBinding(context, "u_vxCounter",
                RHI::BindingResource{{counterbuffer->buffer.get(), 0,
                                      counterbuffer->buffer->size()}});
  updateBinding(context, "u_pCompactIndex",
                RHI::BindingResource{{compactIndices->buffer.get(), 0,
                                      compactIndices->buffer->size()}});

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  // binding resources
  updateBinding(context, "u_Color",
                RHI::BindingResource{color->getUAV(0, 0, 1)});
  updateBinding(context, "u_Debug",
                RHI::BindingResource{debug->getUAV(0, 0, 1)});
  updateBinding(context, "TestDISettingUniform",
                di_test_setting->shared.DITestBinding);
  
  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("FrameIdx");

  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
    int strategy;
  };
  PushConstant pConst = {color->texture->width(), color->texture->height(),
                         batchIdx, strategy};
  //if (invalidDebugPixel > 0) {
  //  pConst.invalidSet = invalidDebugPixel;
  //  invalidDebugPixel = 0;
  //}
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(color->texture->width(), color->texture->height(), 1);

  encoder->end();
}

auto DITestPass::onInteraction(
    Platform::Input* input, Editor::Widget::WidgetInfo* info) noexcept -> void {
  //if (info->isFocused && info->isHovered) {
  //  if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
  //    debugPixel.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
  //    debugPixel.y = std::clamp(info->mousePos.y, 0.f, 719.f);
  //    if (input->isKeyPressed(Platform::SIByL_KEY_SPACE)) {
  //      invalidDebugPixel = 2;
  //    } else {
  //      invalidDebugPixel = 1;
  //    }
  //  }
  //}
}

GITestPass::GITestPass(VXGI::VXGISetting* vx_set)
    : voxel_setting(vx_set) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vxguiding/testgi-compare.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto GITestPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

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
    for (int i = 0; i < 6; ++i)

  reflector.addInput("RadOpaVox6DTex" + std::to_string(i))
      .isTexture()
      .withFormat(RHI::TextureFormat::RG16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT)
                   .setSubresource(0, voxel_setting->clipmapSetting.mip, 0, 1));

  return reflector;
}

auto GITestPass::renderUI() noexcept -> void {
  ImGui::DragInt("Strategy", &strategy);
  // ImGui::DragFloat("Cone Length", &coneLength);
  //{
  //   const char* item_names[] = {"DI Only", "GI Only", "Combined"};
  //   ImGui::Combo("Mode", &showMode, item_names, IM_ARRAYSIZE(item_names),
  //                IM_ARRAYSIZE(item_names));
  // }
}

auto GITestPass::execute(RDG::RenderContext* context,
                         RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* radopaTex[6];
  for (int i = 0; i < 6; ++i)
  radopaTex[i] = renderData.getTexture("RadOpaVox6DTex" + std::to_string(i));

  std::vector<RHI::TextureView*> tex6DArray{
      radopaTex[0]->getSRV(0, voxel_setting->clipmapSetting.mip, 0, 1),
      radopaTex[1]->getSRV(0, voxel_setting->clipmapSetting.mip, 0, 1),
      radopaTex[2]->getSRV(0, voxel_setting->clipmapSetting.mip, 0, 1),
      radopaTex[3]->getSRV(0, voxel_setting->clipmapSetting.mip, 0, 1),
      radopaTex[4]->getSRV(0, voxel_setting->clipmapSetting.mip, 0, 1),
      radopaTex[5]->getSRV(0, voxel_setting->clipmapSetting.mip, 0, 1),
  };

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  // binding resources
  updateBinding(context, "u_Color",
                RHI::BindingResource{color->getUAV(0, 0, 1)});
  updateBinding(context, "uRadOpaVox6D", RHI::BindingResource{tex6DArray});

  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);
  updateBinding(
      context, "uTex3DSampler",
      RHI::BindingResource{voxel_setting->shared.tex3dSampler->sampler.get()});

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("FrameIdx");

  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
    int strategy;
  };
  PushConstant pConst = {color->texture->width(), color->texture->height(),
                         batchIdx, strategy};
  // if (invalidDebugPixel > 0) {
  //   pConst.invalidSet = invalidDebugPixel;
  //   invalidDebugPixel = 0;
  // }
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(color->texture->width(), color->texture->height(), 1);

  encoder->end();
}

ImportInjectPass::ImportInjectPass(VXGI::VXGISetting* vx_set)
    : voxel_setting(vx_set) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vxguiding/vxguiding-debugger.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto ImportInjectPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  uint32_t size = voxel_setting->clipmapSetting.size;
  uint32_t mip = voxel_setting->clipmapSetting.mip;

  reflector.addOutput("ImportTex")
      .isTexture()
      .withSize(Math::ivec3(size, size, size))
      .withLevels(mip)
      .withFormat(RHI::TextureFormat::RG16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR)
                   .setSubresource(0, mip, 0, 1));

  for (int i = 0; i < 6; ++i)
    reflector.addInput("RadOpaVox6DTex" + std::to_string(i))
        .isTexture()
        .withSize(Math::ivec3(size, size, size))
        .withLevels(voxel_setting->clipmapSetting.mip)
        .withFormat(RHI::TextureFormat::RG16_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::StorageBinding}
                .addStage(
                    (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR)
                .setSubresource(0, mip, 0, 1));

  return reflector;
}

auto ImportInjectPass::renderUI() noexcept -> void {
  ImGui::DragFloat("Scaling", &scaling, 0.1, 0, 100);
  ImGui::Checkbox("Use Occlusion", &useOcclusion);
  // ImGui::DragFloat("Cone Length", &coneLength);
  //{
  //   const char* item_names[] = {"DI Only", "GI Only", "Combined"};
  //   ImGui::Combo("Mode", &showMode, item_names, IM_ARRAYSIZE(item_names),
  //                IM_ARRAYSIZE(item_names));
  // }
}

auto ImportInjectPass::onInteraction(
    Platform::Input* input, Editor::Widget::WidgetInfo* info) noexcept -> void {
  if (info->isFocused && info->isHovered) {
    if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
      debug_pixel.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
      debug_pixel.y = std::clamp(info->mousePos.y, 0.f, 719.f);
    }
  }
}

auto ImportInjectPass::execute(RDG::RenderContext* context,
                         RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* importTex = renderData.getTexture("ImportTex");

  
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  // binding resources
  updateBinding(context, "u_ImportanceMap",
                RHI::BindingResource{std::vector<RHI::TextureView*>{
                    importTex->getUAV(0, 0, 1),
                    importTex->getUAV(1, 0, 1),
                    importTex->getUAV(2, 0, 1),
                    importTex->getUAV(3, 0, 1),
                    importTex->getUAV(4, 0, 1),
                    importTex->getUAV(5, 0, 1),
                }});

  GFX::Texture* radopaTex[6];
  for (int i = 0; i < 6; ++i)
    radopaTex[i] = renderData.getTexture("RadOpaVox6DTex" + std::to_string(i));

  updateBinding(
      context, "uRadOpaVox6D",
      RHI::BindingResource{std::vector<RHI::TextureView*>{
          radopaTex[0]->getSRV(0, voxel_setting->clipmapSetting.mip, 0, 1),
          radopaTex[1]->getSRV(0, voxel_setting->clipmapSetting.mip, 0, 1),
          radopaTex[2]->getSRV(0, voxel_setting->clipmapSetting.mip, 0, 1),
          radopaTex[3]->getSRV(0, voxel_setting->clipmapSetting.mip, 0, 1),
          radopaTex[4]->getSRV(0, voxel_setting->clipmapSetting.mip, 0, 1),
          radopaTex[5]->getSRV(0, voxel_setting->clipmapSetting.mip, 0, 1),
      }});

  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);
  updateBinding(
      context, "uTex3DSampler",
      RHI::BindingResource{voxel_setting->shared.tex3dSampler->sampler.get()});

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("FrameIdx");

  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
    float scaling;
    Math::ivec2 debugpixel;
    int use_occlusion;
  };
  PushConstant pConst = {1280, 720, batchIdx, scaling, debug_pixel};
  pConst.use_occlusion = useOcclusion ? 1 : 0;
  // if (invalidDebugPixel > 0) {
  //   pConst.invalidSet = invalidDebugPixel;
  //   invalidDebugPixel = 0;
  //// }
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(64, 64, 64);

  encoder->end();
}

VXClusterComputeInfoPass::VXClusterComputeInfoPass(
    VXGI::VXGISetting* setting)
    : voxel_setting(setting) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "clustering/compute-voxel-info.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VXClusterComputeInfoPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("AABBMin")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("AABBMax")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("CounterBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("CompactIndices")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("Irradiance")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VPLCount")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("VXNormal")
      .isBuffer()
      .withSize(131072 * sizeof(Math::vec4))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("VXPremulIrradiance")
      .isBuffer()
      .withSize(131072 * sizeof(float))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("AssociateBuffer")
      .isBuffer()
      .withSize(131072 * sizeof(uint32_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXClusterComputeInfoPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* aabbMin = renderData.getBuffer("AABBMin");
  GFX::Buffer* aabbMax = renderData.getBuffer("AABBMax");
  GFX::Buffer* counter = renderData.getBuffer("CounterBuffer");
  GFX::Buffer* compactIdx = renderData.getBuffer("CompactIndices");
  GFX::Buffer* vxNormal = renderData.getBuffer("VXNormal");
  GFX::Buffer* vxPremulIrradiance = renderData.getBuffer("VXPremulIrradiance");
  GFX::Buffer* associate = renderData.getBuffer("AssociateBuffer");
  GFX::Texture* irradiance = renderData.getTexture("Irradiance");
  GFX::Texture* vplCount = renderData.getTexture("VPLCount");

  updateBinding(context, "u_vplCount",
                RHI::BindingResource{{vplCount->getUAV(0, 0, 1)}});
  updateBinding(context, "u_pMin",
                RHI::BindingResource{
                    {aabbMin->buffer.get(), 0, aabbMin->buffer->size()}});
  updateBinding(context, "u_associate",
                RHI::BindingResource{
                    {associate->buffer.get(), 0, associate->buffer->size()}});
  updateBinding(context, "u_pMax",
                RHI::BindingResource{
                    {aabbMax->buffer.get(), 0, aabbMax->buffer->size()}});
  updateBinding(context, "u_vxCounter",
                RHI::BindingResource{
                    {counter->buffer.get(), 0, counter->buffer->size()}});
  updateBinding(context, "u_pCompactIndex",
                RHI::BindingResource{
                    {compactIdx->buffer.get(), 0, compactIdx->buffer->size()}});
  updateBinding(context, "u_vxNormal",
                RHI::BindingResource{
                    {vxNormal->buffer.get(), 0, vxNormal->buffer->size()}});
  updateBinding(context, "u_vxIrradiance",
                RHI::BindingResource{{vxPremulIrradiance->buffer.get(), 0,
                                      vxPremulIrradiance->buffer->size()}});
  updateBinding(context, "u_IrradianceMap",
                RHI::BindingResource{{irradiance->getUAV(0, 0, 1)}});

  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroups((131072 + 255) / 256, 1, 1);
  encoder->end();
}

VXClusterInitCenterPass::VXClusterInitCenterPass(
    VXGI::VXGISetting* setting)
    : voxel_setting(setting) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "clustering/init-cluster-centers.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VXClusterInitCenterPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("CounterBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("CompactIndices")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VXNormal")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("ClusterSeeds")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("SVXInfo")
      .isBuffer()
      .withSize(32 * sizeof(Math::vec4) * 2)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("SVXAccumInfo")
      .isBuffer()
      .withSize(32 * sizeof(Math::vec4) * 2)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("GridClusterCount")
      .isTexture()
      .withSize(Math::ivec3(8, 8, 8))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT)
                   .setSubresource(0, 1, 0, 1));
  reflector.addOutput("GridClusterIndices")
      .isTexture()
      .withSize(Math::ivec3(8, 8, 8))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT)
                   .setSubresource(0, 1, 0, 1));
  reflector.addOutput("DispatchIndirectArgs")
      .isBuffer()
      .withSize(8 * sizeof(int))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXClusterInitCenterPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Use Optimized Seed", &use_seed);
}

auto VXClusterInitCenterPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* counter = renderData.getBuffer("CounterBuffer");
  GFX::Buffer* seeds = renderData.getBuffer("ClusterSeeds");
  GFX::Buffer* compactIdx = renderData.getBuffer("CompactIndices");
  GFX::Buffer* vxNormal = renderData.getBuffer("VXNormal");
  GFX::Buffer* svxInfo = renderData.getBuffer("SVXInfo");
  GFX::Buffer* svxAccumInfo = renderData.getBuffer("SVXAccumInfo");
  GFX::Buffer* dispatchIA = renderData.getBuffer("DispatchIndirectArgs");

  GFX::Texture* gcc = renderData.getTexture("GridClusterCount");
  GFX::Texture* gci = renderData.getTexture("GridClusterIndices");
  updateBinding(context, "u_hashingCount",
                RHI::BindingResource{{gcc->getUAV(0, 0, 1)}});
  updateBinding(context, "u_hashingIndices",
                RHI::BindingResource{{gci->getUAV(0, 0, 1)}});

  updateBinding(context, "u_vxCounter",
                RHI::BindingResource{
                    {counter->buffer.get(), 0, counter->buffer->size()}});
  updateBinding(context, "u_seed",
      RHI::BindingResource{{seeds->buffer.get(), 0, seeds->buffer->size()}});
  updateBinding(context, "u_indirectDispatchArgs",
                RHI::BindingResource{
                    {dispatchIA->buffer.get(), 0, dispatchIA->buffer->size()}});
  updateBinding(context, "u_pCompactIndex",
                RHI::BindingResource{
                    {compactIdx->buffer.get(), 0, compactIdx->buffer->size()}});
  updateBinding(context, "u_vxNormal",
                RHI::BindingResource{
                    {vxNormal->buffer.get(), 0, vxNormal->buffer->size()}});
  updateBinding(context, "u_SVoxels",
                RHI::BindingResource{
                    {svxInfo->buffer.get(), 0, svxInfo->buffer->size()}});
  updateBinding(context, "u_accumSVoxels",
                RHI::BindingResource{{svxAccumInfo->buffer.get(), 0,
                                      svxAccumInfo->buffer->size()}});

  struct PushConstant {
    uint32_t batchIdx;
    uint32_t use_seed;
  } pConst = { 
      renderData.getUInt("FrameIdx"),
      use_seed ? 1 :0
  };

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups(1, 1, 1);
  encoder->end();
}

VXClusterSeedingPass::VXClusterSeedingPass() {
  auto comp = GFX::GFXManager::get()->registerShaderModuleResource(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "clustering/kmpp-seeding.spv",
      {nullptr, RHI::ShaderStages::COMPUTE});

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VXClusterSeedingPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("CounterBuffer")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("CompactIndices")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VXNormal")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("ClusterSeeds")
      .isBuffer().withSize(32 * sizeof(int32_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("DebugBuffer")
      .isBuffer().withSize(1024 * sizeof(float) * 4)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXClusterSeedingPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* counter = renderData.getBuffer("CounterBuffer");
  GFX::Buffer* compactIdx = renderData.getBuffer("CompactIndices");
  GFX::Buffer* vxNormal = renderData.getBuffer("VXNormal");
  GFX::Buffer* clusterSeeds = renderData.getBuffer("ClusterSeeds");
  GFX::Buffer* debug = renderData.getBuffer("DebugBuffer");
  
  getBindGroup(context, 0)->updateBinding({
    RHI::BindGroupEntry{0, RHI::BindingResource{{clusterSeeds->buffer.get(), 0, clusterSeeds->buffer->size()}}},
    RHI::BindGroupEntry{1, RHI::BindingResource{{compactIdx->buffer.get(), 0, compactIdx->buffer->size()}}},
    RHI::BindGroupEntry{2, RHI::BindingResource{{vxNormal->buffer.get(), 0, vxNormal->buffer->size()}}},
    RHI::BindGroupEntry{3, RHI::BindingResource{{counter->buffer.get(), 0, counter->buffer->size()}}},
    RHI::BindGroupEntry{4, RHI::BindingResource{{debug->buffer.get(), 0, debug->buffer->size()}}},
  });
    
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroups(1, 1, 1);
  encoder->end();
}

VXClusterFindAssociatePass::VXClusterFindAssociatePass(
    VXGI::VXGISetting* setting)
    : voxel_setting(setting) {
  auto comp = GFX::GFXManager::get()->registerShaderModuleResource(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "clustering/find-center-association.spv",
      {nullptr, RHI::ShaderStages::COMPUTE});

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VXClusterFindAssociatePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("CounterBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("CompactIndices")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VXNormal")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("SVXInfo")
      .isBuffer()
      .withSize(32 * sizeof(Math::vec4) * 2)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("SVXAccumInfo")
      .isBuffer()
      .withSize(32 * sizeof(Math::vec4) * 2)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("GridClusterCount")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT)
                   .setSubresource(0, 1, 0, 1));
  reflector.addInput("GridClusterIndices")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT)
                   .setSubresource(0, 1, 0, 1));
  reflector.addInputOutput("AssociateBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("DispatchIndirectArgs")
      .isBuffer()
      .withSize(3 * sizeof(int))
      .withUsages((uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));
  reflector.addOutput("DebugBUFFER")
      .isBuffer()
      .withSize(8 * sizeof(int) * 131072)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXClusterFindAssociatePass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* gcc = renderData.getTexture("GridClusterCount");
  GFX::Texture* gci = renderData.getTexture("GridClusterIndices");

  GFX::Buffer* associate = renderData.getBuffer("AssociateBuffer");
  GFX::Buffer* compactIdx = renderData.getBuffer("CompactIndices");
  GFX::Buffer* vxNormal = renderData.getBuffer("VXNormal");
  GFX::Buffer* svxInfo = renderData.getBuffer("SVXInfo");
  GFX::Buffer* svxAccumInfo = renderData.getBuffer("SVXAccumInfo");
  GFX::Buffer* counter = renderData.getBuffer("CounterBuffer");
  GFX::Buffer* iargs = renderData.getBuffer("DispatchIndirectArgs");
  GFX::Buffer* DB = renderData.getBuffer("DebugBUFFER");

  getBindGroup(context, 0)->updateBinding({
    RHI::BindGroupEntry{0, RHI::BindingResource{{gcc->getUAV(0, 0, 1)}}},
    RHI::BindGroupEntry{1, RHI::BindingResource{{gci->getUAV(0, 0, 1)}}},
    RHI::BindGroupEntry{2, RHI::BindingResource{{associate->buffer.get(), 0, associate->buffer->size()}}},
    RHI::BindGroupEntry{3, RHI::BindingResource{{compactIdx->buffer.get(), 0, compactIdx->buffer->size()}}},
    RHI::BindGroupEntry{4, RHI::BindingResource{{vxNormal->buffer.get(), 0, vxNormal->buffer->size()}}},
    RHI::BindGroupEntry{5, RHI::BindingResource{{svxInfo->buffer.get(), 0, svxInfo->buffer->size()}}},
    RHI::BindGroupEntry{6, RHI::BindingResource{{svxAccumInfo->buffer.get(), 0, svxAccumInfo->buffer->size()}}},
    RHI::BindGroupEntry{7, RHI::BindingResource{{counter->buffer.get(), 0, counter->buffer->size()}}},
    RHI::BindGroupEntry{8, RHI::BindingResource{{DB->buffer.get(), 0, DB->buffer->size()}}},
  });

  RHI::ComputePassEncoder* encoder = beginPass(context);
  uint32_t pc = noCluster ? 1 : 0;
  encoder->pushConstants(&pc, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(uint32_t));
  encoder->dispatchWorkgroupsIndirect(iargs->buffer.get(), 0);
  encoder->end();
}

auto VXClusterFindAssociatePass::renderUI() noexcept -> void {
  ImGui::Checkbox("No Cluster", &noCluster);
}

VXClusterUpdateCenterPass::VXClusterUpdateCenterPass(VXGI::VXGISetting* setting)
    : voxel_setting(setting) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "clustering/update-cluster-center.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VXClusterUpdateCenterPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("SVXInfo")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("SVXAccumInfo")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("GridClusterCount")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT)
                   .setSubresource(0, 1, 0, 1));
  reflector.addInputOutput("GridClusterIndices")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT)
                   .setSubresource(0, 1, 0, 1));
  return reflector;
}

auto VXClusterUpdateCenterPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* svxInfo = renderData.getBuffer("SVXInfo");
  GFX::Buffer* svxAccumInfo = renderData.getBuffer("SVXAccumInfo");
  GFX::Texture* gcc = renderData.getTexture("GridClusterCount");
  GFX::Texture* gci = renderData.getTexture("GridClusterIndices");
  updateBinding(context, "u_hashingCount",
                RHI::BindingResource{{gcc->getUAV(0, 0, 1)}});
  updateBinding(context, "u_hashingIndices",
                RHI::BindingResource{{gci->getUAV(0, 0, 1)}});
  updateBinding(context, "u_SVoxels",
                RHI::BindingResource{
                    {svxInfo->buffer.get(), 0, svxInfo->buffer->size()}});
  updateBinding(context, "u_accumSVoxels",
                RHI::BindingResource{{svxAccumInfo->buffer.get(), 0,
                                      svxAccumInfo->buffer->size()}});

  uint32_t batchIdx = renderData.getUInt("FrameIdx");

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroups(1, 1, 1);
  encoder->end();
}

VXTreeInternalPass::VXTreeInternalPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "tree/tree-internal-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VXTreeInternalPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("Code")
      .isBuffer().withSize(65536 * sizeof(uint64_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume( RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("Node")
      .isBuffer().withSize(65536 * sizeof(float) * 8)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume( RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("IndirectArgs")
      .isBuffer() .withSize(sizeof(float) * 16)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                  (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess(
                  (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT |
                  (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT |
                        (uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("Descendant")
      .isBuffer().withSize(65536 * sizeof(uint32_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXTreeInternalPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* code = renderData.getBuffer("Code");
  GFX::Buffer* node = renderData.getBuffer("Node");
  GFX::Buffer* descendant = renderData.getBuffer("Descendant");
  GFX::Buffer* indirectArgs = renderData.getBuffer("IndirectArgs");

  updateBinding(
      context, "u_Codes",
      RHI::BindingResource{{code->buffer.get(), 0, code->buffer->size()}});
  updateBinding(
      context, "u_Nodes",
      RHI::BindingResource{{node->buffer.get(), 0, node->buffer->size()}});
  updateBinding(context, "u_ConstrIndirectArgs",
                RHI::BindingResource{{indirectArgs->buffer.get(), 0,
                                      indirectArgs->buffer->size()}});
  updateBinding(context, "u_Descendant",
                RHI::BindingResource{{descendant->buffer.get(), 0, descendant->buffer->size()}});
  
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroupsIndirect(indirectArgs->buffer.get(),
                                      4 * sizeof(uint32_t));
  encoder->end();
}

VXTreeMergePass::VXTreeMergePass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "tree/tree-merge-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VXTreeMergePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("Node")
      .isBuffer().withSize(65536 * sizeof(float) * 8)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume( RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("IndirectArgs")
      .isBuffer().withSize(sizeof(float) * 16)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume( RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("ClusterRoots")
      .isBuffer()
      .withSize(32 * sizeof(int32_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXTreeMergePass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* node = renderData.getBuffer("Node");
  GFX::Buffer* iarg = renderData.getBuffer("IndirectArgs");
  GFX::Buffer* clusterRoot = renderData.getBuffer("ClusterRoots");

  updateBinding(
      context, "u_ConstrIndirectArgs",
      RHI::BindingResource{{iarg->buffer.get(), 0, iarg->buffer->size()}});
  updateBinding(
      context, "u_Nodes",
      RHI::BindingResource{{node->buffer.get(), 0, node->buffer->size()}});
  updateBinding(
      context, "cluster_roots",
                RHI::BindingResource{{clusterRoot->buffer.get(), 0,
                                      clusterRoot->buffer->size()}});

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroupsIndirect(iarg->buffer.get(), 0);
  encoder->end();

  Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
      context->cmdEncoder, RHI::PipelineStages::BOTTOM_OF_PIPE_BIT,
      "tree_building");
}

VXTreeTopLevelPass::VXTreeTopLevelPass() {
  auto comp = GFX::GFXManager::get()->registerShaderModuleResource(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "tree/tree-top-level-constr.spv",
      {nullptr, RHI::ShaderStages::COMPUTE});

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VXTreeTopLevelPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("Node")
      .isBuffer().withSize(65536 * sizeof(float) * 8)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume( RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("ClusterRoots")
      .isBuffer()
      .withSize(32 * sizeof(int32_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("SPixelVisibility")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("TopLevelTree")
      .isBuffer()
      .withSize(64 * sizeof(float) * 40 * 23)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("SPixelAvgVisibility")
    .isBuffer().withSize(40 * 23 * sizeof(uint32_t) * 8)
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  //reflector.addOutput("TopLevelProb")
  //    .isBuffer()
  //    .withSize(32 * sizeof(float) * 40 * 23)
  //    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
  //    .consume(
  //        RDG::BufferInfo::ConsumeEntry{}
  //            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
  //            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXTreeTopLevelPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* node = renderData.getBuffer("Node");
  GFX::Buffer* cr = renderData.getBuffer("ClusterRoots");
  GFX::Buffer* tlt = renderData.getBuffer("TopLevelTree");
  GFX::Buffer* av = renderData.getBuffer("SPixelAvgVisibility");
  //GFX::Buffer* tlp = renderData.getBuffer("TopLevelProb");
  GFX::Texture* spvis = renderData.getTexture("SPixelVisibility");

  getBindGroup(context, 0)->updateBinding({
    RHI::BindGroupEntry{0, RHI::BindingResource{{tlt->buffer.get(), 0, tlt->buffer->size()}}},
    RHI::BindGroupEntry{1, RHI::BindingResource{{cr->buffer.get(), 0, cr->buffer->size()}}},
    RHI::BindGroupEntry{2, RHI::BindingResource{{node->buffer.get(), 0, node->buffer->size()}}},
    RHI::BindGroupEntry{3, RHI::BindingResource{{spvis->getUAV(0, 0, 1)}}},
    RHI::BindGroupEntry{4, RHI::BindingResource{{av->buffer.get(), 0, av->buffer->size()}}},
    //RHI::BindGroupEntry{4, RHI::BindingResource{{tlp->buffer.get(), 0, tlp->buffer->size()}}},
  });

  Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
      context->cmdEncoder, RHI::PipelineStages::TOP_OF_PIPE_BIT, "matrix_tree_build");

  Math::ivec3 map_size = {40, 23, visibility};
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&map_size, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(map_size));
  encoder->dispatchWorkgroups(5, 23, 1);
  encoder->end();

  Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
      context->cmdEncoder, RHI::PipelineStages::BOTTOM_OF_PIPE_BIT,
      "matrix_tree_build");
}

auto VXTreeTopLevelPass::renderUI() noexcept -> void {
  const char* item_names[] = {"Binary", "Average"};
  ImGui::Combo("Sample Mode", &visibility, item_names, IM_ARRAYSIZE(item_names),
               IM_ARRAYSIZE(item_names));
}

VXTreeEncodePass::VXTreeEncodePass(VXGI::VXGISetting* setting)
    : voxel_setting(setting) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "tree/tree-encode-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VXTreeEncodePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("CounterBuffer")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("CompactIndices")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VXClusterAssociation")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("DispatchIndirectArgs")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess(
                       (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));
  reflector.addOutput("Code")
      .isBuffer().withSize(65536 * sizeof(uint64_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume( RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("IndirectArgs")
      .isBuffer().withSize(sizeof(float) * 16)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume( RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXTreeEncodePass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* counter = renderData.getBuffer("CounterBuffer");
  GFX::Buffer* compact = renderData.getBuffer("CompactIndices");
  GFX::Buffer* association = renderData.getBuffer("VXClusterAssociation");
  GFX::Buffer* code = renderData.getBuffer("Code");
  GFX::Buffer* indirectArgs = renderData.getBuffer("IndirectArgs");
  GFX::Buffer* dispatchArgs = renderData.getBuffer("DispatchIndirectArgs");

  updateBinding(context, "u_vxCounter",
                RHI::BindingResource{
                    {counter->buffer.get(), 0, counter->buffer->size()}});
  updateBinding(context, "u_compactIndex",
                RHI::BindingResource{
                    {compact->buffer.get(), 0, compact->buffer->size()}});
  updateBinding(context, "u_clusterIndex",
                RHI::BindingResource{{association->buffer.get(), 0,
                                      association->buffer->size()}});
  updateBinding(
      context, "u_Codes",
      RHI::BindingResource{{code->buffer.get(), 0, code->buffer->size()}});
  updateBinding(context, "u_ConstrIndirectArgs",
                RHI::BindingResource{{indirectArgs->buffer.get(), 0,
                                      indirectArgs->buffer->size()}});
  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);

  Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
      context->cmdEncoder, RHI::PipelineStages::TOP_OF_PIPE_BIT,
      "tree_building");

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroupsIndirect(dispatchArgs->buffer.get(),
                                      4 * sizeof(uint32_t));
  encoder->end();
}

VXTreeIIntializePass::VXTreeIIntializePass(VXGI::VXGISetting* setting)
    : voxel_setting(setting) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "tree/tree-initial-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VXTreeIIntializePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("Irradiance")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("CompactIndices")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VXClusterAssociation")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("Code")
      .isBuffer().withSize(65536 * sizeof(uint64_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume( RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Node")
      .isBuffer().withSize(65536 * sizeof(float) * 12)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume( RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Compact2Leaf")
      .isBuffer().withSize(65536 * sizeof(uint32_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume( RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("ClusterRoots")
      .isBuffer().withSize(32 * sizeof(int32_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume( RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("IndirectArgs")
      .isBuffer().withSize(sizeof(float) * 16)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                  (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume( RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT |
                         (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT |
                        (uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));
  reflector.addOutput("Descendant")
      .isBuffer().withSize(65536 * sizeof(uint32_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXTreeIIntializePass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* irradiance = renderData.getBuffer("Irradiance");
  GFX::Buffer* compact = renderData.getBuffer("CompactIndices");
  GFX::Buffer* association = renderData.getBuffer("VXClusterAssociation");
  GFX::Buffer* code = renderData.getBuffer("Code");
  GFX::Buffer* node = renderData.getBuffer("Node");
  GFX::Buffer* indirectArgs = renderData.getBuffer("IndirectArgs");
  GFX::Buffer* clusterRoots = renderData.getBuffer("ClusterRoots");
  GFX::Buffer* compact2leaf = renderData.getBuffer("Compact2Leaf");
  GFX::Buffer* descendant = renderData.getBuffer("Descendant");

  updateBinding(context, "u_vxIrradiance",
                RHI::BindingResource{
                    {irradiance->buffer.get(), 0, irradiance->buffer->size()}});
  updateBinding(context, "cluster_roots",
                RHI::BindingResource{{clusterRoots->buffer.get(), 0,
                                      clusterRoots->buffer->size()}});
  updateBinding(context, "u_compactIndex",
                RHI::BindingResource{
                    {compact->buffer.get(), 0, compact->buffer->size()}});
  updateBinding(context, "u_clusterIndex",
                RHI::BindingResource{{association->buffer.get(), 0,
                                      association->buffer->size()}});
  updateBinding(context, "compact2leaf",
                RHI::BindingResource{{compact2leaf->buffer.get(), 0,
                                      compact2leaf->buffer->size()}});
  updateBinding(context, "u_Descendant",
                RHI::BindingResource{{descendant->buffer.get(), 0, descendant->buffer->size()}});
  updateBinding(
      context, "u_Codes",
      RHI::BindingResource{{code->buffer.get(), 0, code->buffer->size()}});
  updateBinding(
      context, "u_Nodes",
      RHI::BindingResource{{node->buffer.get(), 0, node->buffer->size()}});
  updateBinding(context, "u_ConstrIndirectArgs",
                RHI::BindingResource{{indirectArgs->buffer.get(), 0,
                                      indirectArgs->buffer->size()}});
  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);
  
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroupsIndirect(indirectArgs->buffer.get(),
                                      8 * sizeof(uint32_t));
  encoder->end();
}

SPixelClearPass::SPixelClearPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "visibility/cvis-clear.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto SPixelClearPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("SPixelVisibility")
      .isTexture()
      .withSize(Math::ivec3(40, 23, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("SPixelAvgVisibility")
      .isBuffer().withSize(40 * 23 * sizeof(uint32_t) * 8)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("SPixelCounter")
      .isTexture()
      .withSize(Math::ivec3(40, 23, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("ClusterCounter")
      .isBuffer().withSize(32 * sizeof(uint32_t))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto SPixelClearPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* spv = renderData.getTexture("SPixelVisibility");
  GFX::Texture* spc = renderData.getTexture("SPixelCounter");
  GFX::Buffer* cc = renderData.getBuffer("ClusterCounter");
  GFX::Buffer* avgvis = renderData.getBuffer("SPixelAvgVisibility");
  updateBinding(context, "u_spixel_visibility",
                RHI::BindingResource{{spv->getUAV(0, 0, 1)}});
  updateBinding(context, "u_spixel_counter",
                RHI::BindingResource{{spc->getUAV(0, 0, 1)}});
  updateBinding(context, "u_cluster_counter",
      RHI::BindingResource{{cc->buffer.get(), 0, cc->buffer->size()}});
  updateBinding(context, "u_spixel_avg_visibility",
      RHI::BindingResource{{avgvis->buffer.get(), 0, avgvis->buffer->size()}});

  Math::ivec2 resolution = {40, 23};
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&resolution,
                         (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(resolution));
  encoder->dispatchWorkgroups((40 + 15) / 16, (23 + 15) / 16, 1);
  encoder->end();
}

SPixelGatherPass::SPixelGatherPass(VXGI::VXGISetting* voxel_setting)
    : voxel_setting(voxel_setting) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "visibility/cvis-assignment.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto SPixelGatherPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("SPixelGathered")
      .isTexture().withSize(Math::ivec3(1280, 736, 1))
      .withFormat(RHI::TextureFormat::RG16_SINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("ClusterGathered")
      .isBuffer().withSize(32 * 32 * 32 * sizeof(float) * 4)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("SPixelVisibility")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("SPixelCounter")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("ClusterCounter")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("SPixelIndexImage")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VPLPositions")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VXInverseIndex")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VXClusterAssociation")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto SPixelGatherPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* spg = renderData.getTexture("SPixelGathered");
  GFX::Texture* spv = renderData.getTexture("SPixelVisibility");
  GFX::Texture* spc = renderData.getTexture("SPixelCounter");
  GFX::Texture* spi = renderData.getTexture("SPixelIndexImage");
  GFX::Texture* pos = renderData.getTexture("VPLPositions");
  GFX::Texture* vxi = renderData.getTexture("VXInverseIndex");
  GFX::Buffer* vxa = renderData.getBuffer("VXClusterAssociation");
  GFX::Buffer* cg = renderData.getBuffer("ClusterGathered");
  GFX::Buffer* cc = renderData.getBuffer("ClusterCounter");

  updateBinding(context, "u_spixel_gathered",
                RHI::BindingResource{{spg->getUAV(0, 0, 1)}});
  updateBinding(context, "u_spixel_visibility",
                RHI::BindingResource{{spv->getUAV(0, 0, 1)}});
  updateBinding(context, "u_spixel_counter",
                RHI::BindingResource{{spc->getUAV(0, 0, 1)}});
  updateBinding(context, "u_spixelIdx",
                RHI::BindingResource{{spi->getUAV(0, 0, 1)}});
  updateBinding(context, "u_vpl_position",
                RHI::BindingResource{{pos->getUAV(0, 0, 1)}});
  updateBinding(context, "u_inverse_index",
                RHI::BindingResource{{vxi->getUAV(0, 0, 1)}});
  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);
  updateBinding(context, "u_associate_buffer",
      RHI::BindingResource{{vxa->buffer.get(), 0, vxa->buffer->size()}});
  updateBinding(context, "u_cluster_gathered",
      RHI::BindingResource{{cg->buffer.get(), 0, cg->buffer->size()}});
  updateBinding(context, "u_cluster_counter",
      RHI::BindingResource{{cc->buffer.get(), 0, cc->buffer->size()}});

  Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
      context->cmdEncoder, RHI::PipelineStages::TOP_OF_PIPE_BIT,
      "matrix_gather");

  struct PushConstant {
    Math::ivec2 mapsize;
  } pConst = {
      {40,23}
  };
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((1280 + 15) / 16, (720 + 15) / 16, 1);
  encoder->end();

  Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
      context->cmdEncoder, RHI::PipelineStages::BOTTOM_OF_PIPE_BIT,
      "matrix_gather");
}

RowColumnPresamplePass::RowColumnPresamplePass(
    VXGI::VXGISetting* voxel_setting)
    : voxel_setting(voxel_setting) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "mrcs/row-column-presample.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto RowColumnPresamplePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("RepresentPixel")
      .isBuffer().withSize(128 * sizeof(float) * 4)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("RepresentVPL")
      .isBuffer().withSize(65536 * sizeof(float) * 4)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("IndirectArgs")
      .isBuffer().withSize(4 * sizeof(uint32_t) * 4)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VXCounter")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VPLPositions")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VXInverseIndex")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume( RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VBuffer")
    .isTexture().withFormat(RHI::TextureFormat::RGBA32_UINT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::StorageBinding}
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("BoundMin")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("BoundMax")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  //reflector.addInputOutput("DirectionalHash")
  //    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
  //    .consume(RDG::BufferInfo::ConsumeEntry{}
  //            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT |
  //                       (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
  //            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto RowColumnPresamplePass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* pos = renderData.getTexture("VPLPositions");
  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* vxi = renderData.getTexture("VXInverseIndex");
  GFX::Buffer* rpix = renderData.getBuffer("RepresentPixel");
  GFX::Buffer* rvpl = renderData.getBuffer("RepresentVPL");
  GFX::Buffer* vxcounter = renderData.getBuffer("VXCounter");
  GFX::Buffer* iarg = renderData.getBuffer("IndirectArgs");

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);

  updateBinding(context, "u_vpl_position",
                RHI::BindingResource{{pos->getUAV(0, 0, 1)}});
  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{{vbuffer->getUAV(0, 0, 1)}});
  updateBinding(context, "u_inverse_index",
                RHI::BindingResource{{vxi->getUAV(0, 0, 1)}});
  updateBinding(context, "VoxelizerUniform",
                voxel_setting->shared.voxUniBinding);
  updateBinding(context, "u_RepresentPixel",
      RHI::BindingResource{{rpix->buffer.get(), 0, rpix->buffer->size()}});
  updateBinding(context, "u_RepresentVPL",
      RHI::BindingResource{{rvpl->buffer.get(), 0, rvpl->buffer->size()}});
  updateBinding(context, "u_vplCounter",
      RHI::BindingResource{{vxcounter->buffer.get(), 0, vxcounter->buffer->size()}});
  updateBinding(context, "u_IndirectArgs",
      RHI::BindingResource{{iarg->buffer.get(), 0, iarg->buffer->size()}});

  GFX::Buffer* bmin = renderData.getBuffer("BoundMin");
  GFX::Buffer* bmax = renderData.getBuffer("BoundMax");
  //GFX::Buffer* dhas = renderData.getBuffer("DirectionalHash");

  updateBindings(context, {
      {"u_BoundMin", RHI::BindingResource{{bmin->buffer.get(), 0, bmin->buffer->size()}}},
      {"u_BoundMax", RHI::BindingResource{{bmax->buffer.get(), 0, bmax->buffer->size()}}},
      //{"u_DirectionHash", RHI::BindingResource{{dhas->buffer.get(), 0, dhas->buffer->size()}}},
      });

  struct PushConstant {
    //Math::ivec2 mapsize;
    Math::ivec2 resolution;
    int rand_seed;
  } pConst = {
      //{40,23},
      {1280,720},
      renderData.getUInt("FrameIdx")
  };
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((1280 + 15) / 16, (720 + 15) / 16, 1);
  encoder->end();
}

VXInfoClearPass::VXInfoClearPass(VXGI::VXGISetting* voxel_setting)
    : voxel_setting(voxel_setting) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "mrcs/vxinfo-clear.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VXInfoClearPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("BoundMin")
      .isBuffer().withSize(65536 * sizeof(uint32_t) * 4)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("BoundMax")
      .isBuffer().withSize(65536 * sizeof(uint32_t) * 4)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  //reflector.addOutput("DirectionalHash")
  //    .isBuffer().withSize(65536 * sizeof(uint32_t) * 2)
  //    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
  //    .consume(RDG::BufferInfo::ConsumeEntry{}
  //            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
  //            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("VXPremulIrradiance")
    .isBuffer().withSize(65536 * sizeof(float))
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("AABBMin")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("AABBMax")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("CounterBuffer")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("CompactIndices")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("Irradiance")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));  
  reflector.addInput("VPLCount")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));  
  return reflector;
}

auto VXInfoClearPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* bmin = renderData.getBuffer("BoundMin");
  GFX::Buffer* bmax = renderData.getBuffer("BoundMax");
  //GFX::Buffer* dhas = renderData.getBuffer("DirectionalHash");
  GFX::Buffer* cont = renderData.getBuffer("CounterBuffer");
  GFX::Buffer* copi = renderData.getBuffer("CompactIndices");
  GFX::Buffer* prem = renderData.getBuffer("VXPremulIrradiance");
  GFX::Buffer* abmin = renderData.getBuffer("AABBMin");
  GFX::Buffer* abmax = renderData.getBuffer("AABBMax");
  GFX::Texture* irra = renderData.getTexture("Irradiance");
  GFX::Texture* vplCount = renderData.getTexture("VPLCount");

  updateBindings(context, {
      {"u_BoundMin", RHI::BindingResource{{bmin->buffer.get(), 0, bmin->buffer->size()}}},
      {"u_BoundMax", RHI::BindingResource{{bmax->buffer.get(), 0, bmax->buffer->size()}}},
      //{"u_DirectionHash", RHI::BindingResource{{dhas->buffer.get(), 0, dhas->buffer->size()}}},
      {"u_PremulIrradiance", RHI::BindingResource{{prem->buffer.get(), 0, prem->buffer->size()}}},
      {"u_vxCounter", RHI::BindingResource{{cont->buffer.get(), 0, cont->buffer->size()}}},
      {"u_CompactIndex", RHI::BindingResource{{copi->buffer.get(), 0, copi->buffer->size()}}},
      {"u_pMin", RHI::BindingResource{{abmin->buffer.get(), 0, abmin->buffer->size()}}},
      {"u_pMax", RHI::BindingResource{{abmax->buffer.get(), 0, abmax->buffer->size()}}},
      {"u_IrradianceMap", RHI::BindingResource{irra->getUAV(0,0,1)}},
      {"u_VPLCount", RHI::BindingResource{{vplCount->getUAV(0, 0, 1)}}},
      {"VoxelizerUniform", voxel_setting->shared.voxUniBinding},
      });

  Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
      context->cmdEncoder, RHI::PipelineStages::TOP_OF_PIPE_BIT, "svoxel_cluster");

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroups(65536 / 512, 1, 1);
  encoder->end();
}

RowVisibilityPass::RowVisibilityPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "mrcs/row-visibility.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto RowVisibilityPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("RepresentVPL")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("RepresentPixel")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("IndirectArgs")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                             (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT |
                        (uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));
  reflector.addOutput("RowVisibility")
    .isBuffer().withSize(65536 * sizeof(uint32_t) * 4)
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto RowVisibilityPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* rpix = renderData.getBuffer("RepresentPixel");
  GFX::Buffer* rvpl = renderData.getBuffer("RepresentVPL");
  GFX::Buffer* iarg = renderData.getBuffer("IndirectArgs");
  GFX::Buffer* rvis = renderData.getBuffer("RowVisibility");

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  updateBinding(context, "u_RepresentPixel",
      RHI::BindingResource{{rpix->buffer.get(), 0, rpix->buffer->size()}});
  updateBinding(context, "u_RepresentVPL",
      RHI::BindingResource{{rvpl->buffer.get(), 0, rvpl->buffer->size()}});
  updateBinding(context, "u_IndirectArgs",
      RHI::BindingResource{{iarg->buffer.get(), 0, iarg->buffer->size()}});
  updateBinding(context, "u_RowVisibility",
      RHI::BindingResource{{rvis->buffer.get(), 0, rvis->buffer->size()}});

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroupsIndirect(iarg->buffer.get(), 2 * 16);
  encoder->end();
}

RowKmppCenterPass::RowKmppCenterPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "mrcs/column-kmpp-seeding.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto RowKmppCenterPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("IndirectArgs")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("RowVisibility")
    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Seed")
    .isBuffer().withSize(32 * sizeof(uint32_t))
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("RowClusterInfo")
    .isBuffer().withSize(32 * sizeof(uint32_t) * 8)
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto RowKmppCenterPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* iarg = renderData.getBuffer("IndirectArgs");
  GFX::Buffer* rvis = renderData.getBuffer("RowVisibility");
  GFX::Buffer* seed = renderData.getBuffer("Seed");
  GFX::Buffer* rcif = renderData.getBuffer("RowClusterInfo");

  updateBindings(context, {
      {"u_IndirectArgs", RHI::BindingResource{{iarg->buffer.get(), 0, iarg->buffer->size()}}},
      {"u_RowVisibility", RHI::BindingResource{{rvis->buffer.get(), 0, rvis->buffer->size()}}},
      {"u_Seeds", RHI::BindingResource{{seed->buffer.get(), 0, seed->buffer->size()}}},
      {"u_RowClusterInfo", RHI::BindingResource{{rcif->buffer.get(), 0, rcif->buffer->size()}}},
      });

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroups(1,1,1);
  encoder->end();
}

RowFindCenterPass::RowFindCenterPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "mrcs/column-find-center.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto RowFindCenterPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("Clusters")
    .isBuffer().withSize(65536 * sizeof(uint32_t))
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("CompactIndices")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("IndirectArgs")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                             (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT |
                        (uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));
  reflector.addInput("RowVisibility")
    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("RowClusterInfo")
    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto RowFindCenterPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* clus = renderData.getBuffer("Clusters");
  GFX::Buffer* comp = renderData.getBuffer("CompactIndices");
  GFX::Buffer* iarg = renderData.getBuffer("IndirectArgs");
  GFX::Buffer* rvis = renderData.getBuffer("RowVisibility");
  GFX::Buffer* rclu = renderData.getBuffer("RowClusterInfo");
  
  updateBindings(context, {
      {"u_Clusters", RHI::BindingResource{{clus->buffer.get(), 0, clus->buffer->size()}}},
      {"u_CompactIndices", RHI::BindingResource{{comp->buffer.get(), 0, comp->buffer->size()}}},
      {"u_IndirectArgs", RHI::BindingResource{{iarg->buffer.get(), 0, iarg->buffer->size()}}},
      {"u_RowVisibility", RHI::BindingResource{{rvis->buffer.get(), 0, rvis->buffer->size()}}},
      {"u_ClusterInfo", RHI::BindingResource{{rclu->buffer.get(), 0, rclu->buffer->size()}}},
      });
    
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroupsIndirect(iarg->buffer.get(), 0);
  encoder->end();

  Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
      context->cmdEncoder, RHI::PipelineStages::BOTTOM_OF_PIPE_BIT, "svoxel_cluster");
}

VXInfoRearrangePass::VXInfoRearrangePass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "mrcs/vxinfo-rearrange.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VXInfoRearrangePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("Compact2Leaf")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("IndirectArgs")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                             (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT |
                        (uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));
  reflector.addInput("BoundMin")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("BoundMax")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  //reflector.addInput("DirectionalHash")
  //    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
  //    .consume(RDG::BufferInfo::ConsumeEntry{}
  //            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
  //            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  //reflector.addOutput("BoundMinOrdered")
  //    .isBuffer().withSize(65536 * sizeof(uint32_t) * 4)
  //    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
  //    .consume(RDG::BufferInfo::ConsumeEntry{}
  //            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
  //            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  //reflector.addOutput("BoundMaxOrdered")
  //    .isBuffer().withSize(65536 * sizeof(uint32_t) * 4)
  //    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
  //    .consume(RDG::BufferInfo::ConsumeEntry{}
  //            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
  //            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  //reflector.addOutput("DirectionalHashOrdered")
  //    .isBuffer().withSize(65536 * sizeof(uint32_t) * 2)
  //    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
  //    .consume(RDG::BufferInfo::ConsumeEntry{}
  //            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
  //            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("AABBMin")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("AABBMax")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VXInfoRearrangePass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* iarg = renderData.getBuffer("IndirectArgs");
  GFX::Buffer* bmin = renderData.getBuffer("BoundMin");
  GFX::Buffer* bmax = renderData.getBuffer("BoundMax");
  //GFX::Buffer* dhas = renderData.getBuffer("DirectionalHash");
  //GFX::Buffer* bmin_ordered = renderData.getBuffer("BoundMinOrdered");
  //GFX::Buffer* bmax_ordered = renderData.getBuffer("BoundMaxOrdered");
  //GFX::Buffer* dhas_ordered = renderData.getBuffer("DirectionalHashOrdered");
  GFX::Buffer* comp2leaf = renderData.getBuffer("Compact2Leaf");
  GFX::Buffer* abmin = renderData.getBuffer("AABBMin");
  GFX::Buffer* abmax = renderData.getBuffer("AABBMax");

  updateBindings(context, {
      {"u_BoundMin", RHI::BindingResource{{bmin->buffer.get(), 0, bmin->buffer->size()}}},
      {"u_BoundMax", RHI::BindingResource{{bmax->buffer.get(), 0, bmax->buffer->size()}}},
      //{"u_DirectionHash", RHI::BindingResource{{dhas->buffer.get(), 0, dhas->buffer->size()}}},
      //{"u_BoundMinOrdered", RHI::BindingResource{{bmin_ordered->buffer.get(), 0, bmin_ordered->buffer->size()}}},
      //{"u_BoundMaxOrdered", RHI::BindingResource{{bmax_ordered->buffer.get(), 0, bmax_ordered->buffer->size()}}},
      //{"u_DirectionHashOrdered", RHI::BindingResource{{dhas_ordered->buffer.get(), 0, dhas_ordered->buffer->size()}}},
      {"u_pMin", RHI::BindingResource{{abmin->buffer.get(), 0, abmin->buffer->size()}}},
      {"u_pMax", RHI::BindingResource{{abmax->buffer.get(), 0, abmax->buffer->size()}}},
      {"u_IndirectArgs", RHI::BindingResource{{iarg->buffer.get(), 0, iarg->buffer->size()}}},
      {"u_compact2leaf", RHI::BindingResource{{comp2leaf->buffer.get(), 0, comp2leaf->buffer->size()}}},
      });

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&threshold_use_vpl_bound, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(threshold_use_vpl_bound));
  encoder->dispatchWorkgroupsIndirect(iarg->buffer.get(), 0);
  encoder->end();
}

auto VXInfoRearrangePass::renderUI() noexcept -> void {
  ImGui::SliderInt("Label", &threshold_use_vpl_bound, -1, 100);
}

SPixelVisibilityPass::SPixelVisibilityPass() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "visibility/cvis-visibility-check.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto SPixelVisibilityPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("SPixelGathered")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("ClusterGathered")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("SPixelCounter")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("ClusterCounter")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VBuffer")
      .isTexture().withFormat(RHI::TextureFormat::RGBA32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("SPixelVisibility")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Debug")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto SPixelVisibilityPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Addition Check", &do_execute);
}

auto SPixelVisibilityPass::execute(RDG::RenderContext* context,
                               RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* spg = renderData.getTexture("SPixelGathered");
  GFX::Texture* spv = renderData.getTexture("SPixelVisibility");
  GFX::Texture* spc = renderData.getTexture("SPixelCounter");
  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* debug = renderData.getTexture("Debug");
  GFX::Buffer* cg = renderData.getBuffer("ClusterGathered");
  GFX::Buffer* cc = renderData.getBuffer("ClusterCounter");

  updateBinding(context, "u_spixel_gathered",
                RHI::BindingResource{{spg->getUAV(0, 0, 1)}});
  updateBinding(context, "u_spixel_visibility",
                RHI::BindingResource{{spv->getUAV(0, 0, 1)}});
  updateBinding(context, "u_spixel_counter",
                RHI::BindingResource{{spc->getUAV(0, 0, 1)}});
  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{{vbuffer->getUAV(0, 0, 1)}});
  updateBinding(context, "u_Debug",
                RHI::BindingResource{{debug->getUAV(0, 0, 1)}});
  updateBinding(
      context, "u_cluster_gathered",
      RHI::BindingResource{{cg->buffer.get(), 0, cg->buffer->size()}});
  updateBinding(
      context, "u_cluster_counter",
      RHI::BindingResource{{cc->buffer.get(), 0, cc->buffer->size()}});


  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("FrameIdx");

  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
  };
  PushConstant pConst = {1280, 720, batchIdx};
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  if (do_execute) encoder->traceRays(1280, 736, 1);
  encoder->end();
}

SPixelVisibilityEXPass::SPixelVisibilityEXPass(VXGI::VXGISetting* vx_set)
    : voxel_setting(vx_set) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxguiding/"
      "visibility/cvis-visibility-check-comp.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto SPixelVisibilityEXPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("SPixelGathered")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("ClusterGathered")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("SPixelCounter")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("ClusterCounter")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VBuffer")
      .isTexture().withFormat(RHI::TextureFormat::RGBA32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("SPixelVisibility")
      .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Debug")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
   reflector.addInputOutput("SPixelAvgVisibility")
       .isBuffer().withSize(40 * 23 * sizeof(uint32_t) * 8)
       .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
       .consume(RDG::BufferInfo::ConsumeEntry{}
               .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
               .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  //reflector.addInput("BoundMinOrdered")
  //    .isBuffer().withSize(65536 * sizeof(uint32_t) * 4)
  //    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
  //    .consume(RDG::BufferInfo::ConsumeEntry{}
  //             .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
  //            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  //reflector.addInput("BoundMaxOrdered")
  //    .isBuffer().withSize(65536 * sizeof(uint32_t) * 4)
  //    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
  //    .consume(RDG::BufferInfo::ConsumeEntry{}
  //            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
  //            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  //reflector.addInput("DirectionalHashOrdered")
  //    .isBuffer().withSize(65536 * sizeof(uint32_t) * 2)
  //    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
  //    .consume(RDG::BufferInfo::ConsumeEntry{}
  //            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
  //            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  //reflector.addInput("Descendant")
  //    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
  //    .consume(RDG::BufferInfo::ConsumeEntry{}
  //            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
  //            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  //reflector.addInput("ClusterRoots")
  //    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
  //    .consume(RDG::BufferInfo::ConsumeEntry{}
  //            .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
  //            .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto SPixelVisibilityEXPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Addition Check", &do_execute);
}

auto SPixelVisibilityEXPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* spg = renderData.getTexture("SPixelGathered");
  GFX::Texture* spv = renderData.getTexture("SPixelVisibility");
  GFX::Texture* spc = renderData.getTexture("SPixelCounter");
  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* debug = renderData.getTexture("Debug");
  GFX::Buffer* cg = renderData.getBuffer("ClusterGathered");
  GFX::Buffer* cc = renderData.getBuffer("ClusterCounter");
  GFX::Buffer* av = renderData.getBuffer("SPixelAvgVisibility");

  //GFX::Buffer* bmio = renderData.getBuffer("BoundMinOrdered");
  //GFX::Buffer* bmao = renderData.getBuffer("BoundMaxOrdered");
  //GFX::Buffer* dhao = renderData.getBuffer("DirectionalHashOrdered");
  //GFX::Buffer* desc = renderData.getBuffer("Descendant");
  //GFX::Buffer* croo = renderData.getBuffer("ClusterRoots");
  //
  //updateBindings(context, {
  //    {"u_BoundMin", RHI::BindingResource{{bmio->buffer.get(), 0, bmio->buffer->size()}}},
  //    {"u_BoundMax", RHI::BindingResource{{bmao->buffer.get(), 0, bmao->buffer->size()}}},
  //    {"u_DirectionHash", RHI::BindingResource{{dhao->buffer.get(), 0, dhao->buffer->size()}}},
  //    {"u_Descendant", RHI::BindingResource{{desc->buffer.get(), 0, desc->buffer->size()}}},
  //    {"u_ClusterRoots", RHI::BindingResource{{croo->buffer.get(), 0, croo->buffer->size()}}},
  //    {"VoxelizerUniform", voxel_setting->shared.voxUniBinding},
  //    });

  updateBinding(context, "u_spixel_gathered",
                RHI::BindingResource{{spg->getUAV(0, 0, 1)}});
  updateBinding(context, "u_spixel_visibility",
                RHI::BindingResource{{spv->getUAV(0, 0, 1)}});
  updateBinding(context, "u_spixel_counter",
                RHI::BindingResource{{spc->getUAV(0, 0, 1)}});
  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{{vbuffer->getUAV(0, 0, 1)}});
  updateBinding(context, "u_Debug",
                RHI::BindingResource{{debug->getUAV(0, 0, 1)}});
  updateBinding(
      context, "u_cluster_gathered",
      RHI::BindingResource{{cg->buffer.get(), 0, cg->buffer->size()}});
  updateBinding(
      context, "u_cluster_counter",
      RHI::BindingResource{{cc->buffer.get(), 0, cc->buffer->size()}});
  updateBinding(
      context, "u_spixel_avg_visibility",
      RHI::BindingResource{{av->buffer.get(), 0, av->buffer->size()}});


  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
      context->cmdEncoder, RHI::PipelineStages::TOP_OF_PIPE_BIT, "matrix_rt");

  RHI::ComputePassEncoder* encoder = beginPass(context);
  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t map_width;
    uint32_t map_height;
    uint32_t sample_batch;
  };
  PushConstant pConst = {1280, 720, 40, 23, renderData.getUInt("FrameIdx")};
  if (do_execute) {
    encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                           sizeof(PushConstant));
    encoder->dispatchWorkgroups(1280 / 32, 736 / 8, 1);
  }
  encoder->end();

  Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
      context->cmdEncoder, RHI::PipelineStages::BOTTOM_OF_PIPE_BIT,
      "matrix_rt");
}

VPLVisualizePass::VPLVisualizePass() {
  auto [vert, frag] = GFX::ShaderLoader_SLANG::load<2u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vxguiding/visibility/vpl-visualize.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 2>{
          std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto VPLVisualizePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA8_UNORM)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(0));
  reflector.addInternal("Depth")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::DepthStencilAttachment}
                   .enableDepthWrite(true)
                   .setAttachmentLoc(0)
                   .setDepthCompareFn(RHI::CompareFunction::LESS));
  reflector.addInput("ClusterGathered")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT));
  reflector.addInput("ClusterCounter")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT));
  return reflector;
}

auto VPLVisualizePass::execute(RDG::RenderContext* context,
                               RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* depth = renderData.getTexture("Depth");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{color->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{
          depth->getDSV(0, 0, 1), 1, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE,
          false, 0, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE, false},
  };

  GFX::Buffer* cg = renderData.getBuffer("ClusterGathered");
  GFX::Buffer* cc = renderData.getBuffer("ClusterCounter");
  updateBinding(
      context, "u_cluster_gathered",
      RHI::BindingResource{{cg->buffer.get(), 0, cg->buffer->size()}});
  updateBinding(
      context, "u_cluster_counter",
      RHI::BindingResource{{cc->buffer.get(), 0, cc->buffer->size()}});

  updateBinding(context, "CameraBuffer",
                renderData.getBindingResource("GlobalUniforms").value());

  RHI::RenderPassEncoder* encoder = beginPass(context, color);
  encoder->pushConstants(&pConst,
                         (uint32_t)RHI::ShaderStages::VERTEX,
                         0, sizeof(PushConstantBuffer));

  encoder->draw(30, 1024 * 32, 0, 0);
  encoder->end();
}

auto VPLVisualizePass::renderUI() noexcept -> void {
  ImGui::DragFloat("Size Scale", &pConst.gVPLRenderScale, 0.01f, 0);
}

}