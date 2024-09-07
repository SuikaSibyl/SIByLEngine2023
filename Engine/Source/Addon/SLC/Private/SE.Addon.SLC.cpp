#include "../Public/SE.Addon.SLC.hpp"
#include "../Public/SE.Addon.VBuffer.hpp"
#include <SE.Addon.Postprocess.hpp>

namespace SIByL::Addon::SLC {
MortonCodePass::MortonCodePass(VPL::VPLSpawnInfo* info) : vplInfo(info) {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "slc/gen-morton-code.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  // aligned size
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto MortonCodePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  uint32_t alignedSize = ((vplInfo->maxNumber*vplInfo->maxDepth * sizeof(uint64_t) + 63) / 64) * 64;
  reflector.addOutput("KeyIndexBuffer")
      .isBuffer()
      .withSize(alignedSize)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("CounterBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("VPLPositions")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto MortonCodePass::execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  const uint32_t dispatch_size = (vplInfo->maxNumber*vplInfo->maxDepth + 511) / 512;
  GFX::Buffer* vp = renderData.getBuffer("VPLPositions");
  GFX::Buffer* ki = renderData.getBuffer("KeyIndexBuffer");
  GFX::Buffer* cb = renderData.getBuffer("CounterBuffer");

  updateBinding(
      context, "u_vplPositions",
      RHI::BindingResource{{vp->buffer.get(), 0, vp->buffer->size()}});
  updateBinding(
      context, "u_vplCounter",
      RHI::BindingResource{{cb->buffer.get(), 0, cb->buffer->size()}});
  updateBinding(
      context, "u_keyIndexList",
      RHI::BindingResource{{ki->buffer.get(), 0, ki->buffer->size()}});

  struct PushConstant {
    Math::vec3 pMin;
    int padding;
    Math::vec3 pMax;
  } pConst;
  const Math::bounds3 aabb = RACommon::get()->sceneAABB;
  pConst.pMin = aabb.pMin;
  pConst.pMax = aabb.pMax;

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups(dispatch_size, 1, 1);
  encoder->end();
}

struct SLCNode {
  Math::vec3 boundMin;
  float intensity;
  Math::vec3 boundMax;
  int ID;
  Math::vec4 cone;  // xyz cone axis, w cone angle
};

inline int CalculateTreeLevels(int numVPLs) {
  return int(std::ceil(std::log2(numVPLs))) + 1;
}

GenLevel0Pass::GenLevel0Pass(VPL::VPLSpawnInfo* info) : vplInfo(info) {
  // compute nearest power of two
  numTreeLevels = CalculateTreeLevels(info->maxNumber);
  numTreeLights = 1 << (numTreeLevels - 1);
  numStorageNodes = 2 * numTreeLights;

  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "slc/gen-level-0-from-lights.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  // aligned size
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto GenLevel0Pass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("VPLPositions")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VPLNormals")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("VPLColors")
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
  reflector.addInput("KeyIndexBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("NodesBuffer")
      .isBuffer()
      .withSize(numStorageNodes * sizeof(SLCNode))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto GenLevel0Pass::execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  const uint32_t dispatch_size = (vplInfo->maxNumber*vplInfo->maxDepth + 511) / 512;
  GFX::Buffer* ki = renderData.getBuffer("KeyIndexBuffer");
  GFX::Buffer* nb = renderData.getBuffer("NodesBuffer");

  VPL::DVPLPack dvpl_pack = VPL::DVPLPack::fetchPack(renderData);
  dvpl_pack.bindPack(this, context);

  updateBinding(
      context, "u_keyIndexList",
      RHI::BindingResource{{ki->buffer.get(), 0, ki->buffer->size()}});
  updateBinding(
      context, "u_nodes",
      RHI::BindingResource{{nb->buffer.get(), 0, nb->buffer->size()}});

  RHI::ComputePassEncoder* encoder = beginPass(context);
  Math::uvec2 pConst = {numTreeLevels, numTreeLights};
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(Math::uvec2));
  encoder->dispatchWorkgroups(dispatch_size, 1, 1);
  encoder->end();
}

GenLevelInternalPass::GenLevelInternalPass(ParameterSet const& para)
    : para(para) {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "slc/gen-level-from-level.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  // aligned size
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto GenLevelInternalPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("NodesBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto GenLevelInternalPass::execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  const uint32_t dispatch_size = (para.numDstLevelsLights + 511) / 512;
  GFX::Buffer* nb = renderData.getBuffer("NodesBuffer");

  updateBinding(
      context, "u_nodes",
      RHI::BindingResource{{nb->buffer.get(), 0, nb->buffer->size()}});

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&para, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(ParameterSet));
  encoder->dispatchWorkgroups(dispatch_size, 1, 1);
  encoder->end();
}

SLCVisualizePass::SLCVisualizePass(VPL::VPLSpawnInfo* info) : info(info) {
  auto [vert, frag] = GFX::ShaderLoader_SLANG::load<2u>(
      "../Engine/Shaders/SRenderer/addon/"
      "slc/slc-visualizer.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 2>{
          std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));

  int numTreeLevels = CalculateTreeLevels(info->maxNumber);
  numTreeLights = 1 << (numTreeLevels - 1);
}

auto SLCVisualizePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("NodesBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
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
  return reflector;
}

auto SLCVisualizePass::execute(RDG::RenderContext* context,
                               RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* depth = renderData.getTexture("Depth");
  GFX::Buffer* nodes = renderData.getBuffer("NodesBuffer");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{color->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::LOAD,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{
          depth->getDSV(0, 0, 1), 1, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE,
          false, 0, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE, false},
  };

  updateBinding(context, "CameraBuffer",
                renderData.getBindingResource("GlobalUniforms").value());
  updateBinding(
      context, "u_nodes",
      RHI::BindingResource{{nodes->buffer.get(), 0, nodes->buffer->size()}});

  pConst.resolution.x = color->texture->width();
  pConst.resolution.y = color->texture->height();

  RHI::RenderPassEncoder* encoder = beginPass(context, color);
  encoder->pushConstants(&pConst,
                         (uint32_t)RHI::ShaderStages::VERTEX,
                         0, sizeof(PushConstantBuffer));

  encoder->draw(360, numTreeLights, 0, 0);
  encoder->end();
}

auto SLCVisualizePass::renderUI() noexcept -> void {
  ImGui::DragFloat("Line Width", &pConst.line_width, 0.01f, 0);
  ImGui::DragInt("Show Level", &pConst.showLevel, 1);
}
SLCGIPass::SLCGIPass(VPL::VPLSpawnInfo* info) : info(info) {
  int numTreeLevels = CalculateTreeLevels(info->maxNumber);
  leafStartIndex = 1 << (numTreeLevels - 1);

  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "slc/slc-vpl-gi.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto SLCGIPass::reflect() noexcept -> RDG::PassReflection {
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
  reflector.addInput("VPLPositions")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("VPLNormals")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("VPLColors")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("NodesBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto SLCGIPass::execute(RDG::RenderContext* context,
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
  GFX::Texture* debug = renderData.getTexture("Debug");
  GFX::Buffer* pb = renderData.getBuffer("VPLPositions");
  GFX::Buffer* nb = renderData.getBuffer("VPLNormals");
  GFX::Buffer* cb = renderData.getBuffer("VPLColors");
  GFX::Buffer* nodes = renderData.getBuffer("NodesBuffer");

  updateBinding(context, "u_vBuffer",
                RHI::BindingResource{{vbuffer->getUAV(0, 0, 1)}});
  updateBinding(context, "u_color",
                RHI::BindingResource{{color->getUAV(0, 0, 1)}});
  updateBinding(context, "u_debug",
                RHI::BindingResource{{debug->getUAV(0, 0, 1)}});
  updateBinding(
      context, "u_vplPositions",
      RHI::BindingResource{{pb->buffer.get(), 0, pb->buffer->size()}});
  updateBinding(
      context, "u_vplNormals",
      RHI::BindingResource{{nb->buffer.get(), 0, nb->buffer->size()}});
  updateBinding(
      context, "u_vplColors",
      RHI::BindingResource{{cb->buffer.get(), 0, cb->buffer->size()}});
  updateBinding(
      context, "u_nodeBuffer",
      RHI::BindingResource{{nodes->buffer.get(), 0, nodes->buffer->size()}});

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  
  const Math::bounds3 aabb = RACommon::get()->sceneAABB;
  const Math::vec3 center = 0.5 * (aabb.pMax + aabb.pMin);
  const float radius = 0.5 * Math::length(aabb.pMax - aabb.pMin);

  uint32_t slc_config = distanceType << 2;
  if (useApproximateGoemetry) slc_config += 1;
  if (useLightCone) slc_config += 2;

  struct PushConstant {
    Math::uvec2 resolution;
    uint32_t sample_batch;
    uint32_t leaf_start_index;
    Math::vec4 bounding_sphere;
    float inv_vpl_paths;
    uint32_t slc_config;
    int spp;
  } pConst = {{vbuffer->texture->width(), vbuffer->texture->height()},
              renderData.getUInt("FrameIdx"),
              leafStartIndex,
              Math::vec4{center, radius},
              1.f / info->maxNumber,
              slc_config,
              spp};

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(pConst.resolution.x, pConst.resolution.y, 1);

  encoder->end();
}

auto SLCGIPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Use ApproxGeom", &useApproximateGoemetry);
  ImGui::Checkbox("Use Lightcone", &useLightCone);
  ImGui::DragInt("Distance Type", &distanceType, 1, 0, 2);
  ImGui::DragInt("SPP", &spp, 1, 1, 5);
}

SLCBuildGraph::SLCBuildGraph(VPL::VPLSpawnInfo* spawn_info)
    : spawn_info(spawn_info) {

}

auto SLCBuildGraph::alias() noexcept -> RDG::AliasDict {
  RDG::AliasDict dict;
  dict.addAlias("CounterBuffer", CONCAT("MortonCode Pass"), "CounterBuffer");
  dict.addAlias("VPLPositions", CONCAT("MortonCode Pass"), "VPLPositions");

  return dict;
}

auto SLCBuildGraph::onRegister(RDG::Graph* graph) noexcept -> void {
  //// moton coding pass
  //graph->addPass(std::make_unique<MortonCodePass>(&spawn_info),
  //               CONCAT("MortonCode Pass"));
  //// sort pass
  //graph->addSubgraph(
  //    std::make_unique<Addon::BitonicSort::BitonicSort>(&sort_info),
  //    CONCAT("SortMorton Pass"));
  //graph->addEdge(CONCAT("MortonCode Pass"), "KeyIndexBuffer",
  //               CONCAT("SortMorton Pass"), "Input");
  //graph->addEdge("VPLVisualize Pass", "CounterBuffer", "SortMorton Pass",
  //        "CounterBuffer");
  //// gen level 0 pass
  //addPass(std::make_unique<GenLevel0Pass>(&spawn_info), "GenLevel0 Pass");
  //VPL::DVPLPack::addEdge("VPLSpawn Pass", "GenLevel0 Pass", this);
  //addEdge("SortMorton Pass", "Output", "GenLevel0 Pass", "KeyIndexBuffer");

  //const int numLevels = CalculateTreeLevels(spawn_info.maxNumber);
  //const int maxWorkLoad = 2048;
  //int srcLevel = 0;
  //int internamPassId = 0;
  //std::string thisName;
  //std::string prevName =
  //    "GenLevel" + std::to_string(internamPassId++) + " Pass";

  //for (int dstLevelStart = 1; dstLevelStart < numLevels;) {
  //  int dstLevelEnd;
  //  int workLoad = 0;
  //  for (dstLevelEnd = dstLevelStart + 1; dstLevelEnd < numLevels;
  //       dstLevelEnd++) {
  //    workLoad += 1 << (numLevels - 1 - srcLevel);
  //    if (workLoad > maxWorkLoad) break;
  //  }

  //  GenLevelInternalPass::ParameterSet constants;
  //  constants.srcLevel = srcLevel;
  //  constants.dstLevelStart = dstLevelStart;
  //  constants.dstLevelEnd = dstLevelEnd;
  //  constants.numLevels = numLevels;
  //  constants.numDstLevelsLights =
  //      (1 << (numLevels - dstLevelStart)) - (1 << (numLevels - dstLevelEnd));

  //  thisName = "GenLevel" + std::to_string(internamPassId++) + " Pass";
  //  addPass(std::make_unique<GenLevelInternalPass>(constants), thisName);
  //  addEdge(prevName, "NodesBuffer", thisName, "NodesBuffer");
  //  prevName = thisName;

  //  srcLevel = dstLevelEnd - 1;
  //  dstLevelStart = dstLevelEnd;
  //}
}

SLCTestGraph::SLCTestGraph() {
  // spawn info
  spawn_info.maxDepth = 2;
  spawn_info.maxNumber = 1024 * 1024;
  // sort setting config
  sort_info.element_count = spawn_info.maxNumber * spawn_info.maxDepth;
  sort_info.type = BitonicSort::BitonicSortSetting::ElementType::UINT64;
  sort_info.dispath = BitonicSort::BitonicSortSetting::DispathType::DYNAMIC_INDIRECT;

  // counter invalid pass
  addPass(std::make_unique<Addon::VPL::CounterInvalidPass>(&spawn_info), "CounterInvalid Pass");
  // spawn pass
  addPass(std::make_unique<Addon::VPL::VPLSpawnPass>(&spawn_info), "VPLSpawn Pass");
  addEdge("CounterInvalid Pass", "CounterBuffer", "VPLSpawn Pass", "CounterBuffer");
  // visualize pass
  //addPass(std::make_unique<Addon::VPL::VPLVisualizePass>(&spawn_info),
  //        "VPLVisualize Pass");
  //VPL::DVPLPack::addEdge("VPLSpawn Pass", "VPLVisualize Pass", this);

  // moton coding pass
  addPass(std::make_unique<MortonCodePass>(&spawn_info), "MortonCode Pass");
  addEdge("VPLSpawn Pass", "CounterBuffer", "MortonCode Pass",
          "CounterBuffer");
  addEdge("VPLSpawn Pass", "VPLPositions", "MortonCode Pass",
          "VPLPositions");
  // sort pass
  addSubgraph(std::make_unique<Addon::BitonicSort::BitonicSort>(&sort_info),
              "SortMorton Pass");
  addEdge("MortonCode Pass", "KeyIndexBuffer", "SortMorton Pass", "Input");
  addEdge("VPLSpawn Pass", "CounterBuffer", "SortMorton Pass",
          "CounterBuffer");
  // gen level 0 pass
  addPass(std::make_unique<GenLevel0Pass>(&spawn_info), "GenLevel0 Pass");
  VPL::DVPLPack::addEdge("VPLSpawn Pass", "GenLevel0 Pass", this);
  addEdge("SortMorton Pass", "Output", "GenLevel0 Pass", "KeyIndexBuffer");

  const int numLevels = CalculateTreeLevels(spawn_info.maxNumber);
  const int maxWorkLoad = 2048;
  int srcLevel = 0;
  int internamPassId = 0;
  std::string thisName;
  std::string prevName =
      "GenLevel" + std::to_string(internamPassId++) + " Pass";

  for (int dstLevelStart = 1; dstLevelStart < numLevels;) {
    int dstLevelEnd;
    int workLoad = 0;
    for (dstLevelEnd = dstLevelStart + 1; dstLevelEnd < numLevels;
         dstLevelEnd++) {
      workLoad += 1 << (numLevels - 1 - srcLevel);
      if (workLoad > maxWorkLoad) break;
    }

    GenLevelInternalPass::ParameterSet constants;
    constants.srcLevel = srcLevel;
    constants.dstLevelStart = dstLevelStart;
    constants.dstLevelEnd = dstLevelEnd;
    constants.numLevels = numLevels;
    constants.numDstLevelsLights =
        (1 << (numLevels - dstLevelStart)) - (1 << (numLevels - dstLevelEnd));

    thisName = "GenLevel" + std::to_string(internamPassId++) + " Pass";
    addPass(std::make_unique<GenLevelInternalPass>(constants), thisName);
    addEdge(prevName, "NodesBuffer", thisName, "NodesBuffer");
    prevName = thisName;

    srcLevel = dstLevelEnd - 1;
    dstLevelStart = dstLevelEnd;
  }

  // visualize SLC
  bool visualize_slc = false;
  if (visualize_slc) {
    addPass(std::make_unique<SLCVisualizePass>(&spawn_info), "VisSLC Pass");
    addEdge(prevName, "NodesBuffer", "VisSLC Pass", "NodesBuffer");
    addEdge("VPLSpawn Pass", "Color", "VisSLC Pass", "Color");  
  }

  // VBuffer Pass
  addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(), "VBuffer Pass");
  // SLC GI
  addPass(std::make_unique<SLCGIPass>(&spawn_info), "SLC GI Pass");
  addEdge(prevName, "NodesBuffer", "SLC GI Pass", "NodesBuffer");
  addEdge("VPLSpawn Pass", "VPLPositions", "SLC GI Pass", "VPLPositions");
  addEdge("VPLSpawn Pass", "VPLNormals", "SLC GI Pass", "VPLNormals");
  addEdge("VPLSpawn Pass", "VPLColors", "SLC GI Pass", "VPLColors");
  addEdge("VBuffer Pass", "VBuffer", "SLC GI Pass", "VBuffer");

  addPass(std::make_unique<Postprocess::AccumulatePass>(), "Accumulation Pass");
  addEdge("SLC GI Pass", "Color", "Accumulation Pass", "Input");
  
  //addPass(std::make_unique<Addon::Postprocess::ToneMapperPass>(), "ToneMapper Pass");
  //addEdge("Accumulation Pass", "Output", "ToneMapper Pass", "Input");

  markOutput("Accumulation Pass", "Output");
  //markOutput("ToneMapper Pass", "Output");
  //markOutput("TestVBuffer Pass", "Color");
}
}