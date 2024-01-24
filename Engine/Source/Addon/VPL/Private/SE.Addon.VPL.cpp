#include "../Public/SE.Addon.VPL.hpp"

namespace SIByL::Addon::VPL {
CounterInvalidPass::CounterInvalidPass(VPLSpawnInfo* info) : info(info) { 
    pReflection = reflect(); 
}

auto CounterInvalidPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
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

auto CounterInvalidPass::execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  if (info->respawn == false) return;
  // Just fill buffer with 0
  int32_t zero = 0;
  const float& fzero = reinterpret_cast<const float&>(zero);
  GFX::Buffer* buffer = renderData.getBuffer("CounterBuffer");
  context->cmdEncoder->fillBuffer(buffer->buffer.get(), 0, 64, fzero);
}

auto DVPLPack::fetchPack(RDG::RenderData const& renderData) noexcept
    -> DVPLPack {
  DVPLPack pack;
  pack.pb = renderData.getBuffer("VPLPositions");
  pack.nb = renderData.getBuffer("VPLNormals");
  pack.cb = renderData.getBuffer("VPLColors");
  pack.counter = renderData.getBuffer("CounterBuffer");
  return pack;
}

auto DVPLPack::addEdge(std::string const& src, std::string const& tgt,
    RDG::Graph* graph) noexcept -> void {
  graph->addEdge(src, "VPLPositions", tgt, "VPLPositions");
  graph->addEdge(src, "VPLNormals", tgt, "VPLNormals");
  graph->addEdge(src, "VPLColors", tgt, "VPLColors");
  graph->addEdge(src, "CounterBuffer", tgt, "CounterBuffer");
}

auto DVPLPack::bindPack(RDG::PipelinePass* pass,
                        RDG::RenderContext* context) noexcept -> void {
  pass->updateBinding(
      context, "u_vplPositions",
      RHI::BindingResource{{pb->buffer.get(), 0, pb->buffer->size()}});
  pass->updateBinding(
      context, "u_vplNormals",
      RHI::BindingResource{{nb->buffer.get(), 0, nb->buffer->size()}});
  pass->updateBinding(
      context, "u_vplColors",
      RHI::BindingResource{{cb->buffer.get(), 0, cb->buffer->size()}});
  pass->updateBinding(context, "u_vplCounter",
                RHI::BindingResource{
                    {counter->buffer.get(), 0, counter->buffer->size()}});
}

VPLSpawnPass::VPLSpawnPass(VPLSpawnInfo* info) : info(info) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vpl/vpl-spawn.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto VPLSpawnPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addOutput("VPLPositions")
      .isBuffer().withSize(info->maxNumber * info->maxDepth * sizeof(Math::vec4))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("VPLNormals")
      .isBuffer().withSize(info->maxNumber * info->maxDepth * sizeof(Math::vec4))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("VPLColors")
      .isBuffer().withSize(info->maxNumber * info->maxDepth * sizeof(Math::vec4))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInputOutput("CounterBuffer")
      .isBuffer().withSize(64) // potential alignment requirement, could be waste of memory
      // as we only need to use a single int type here actually ...
      // But for clear meaning, we just alloc this explicitly for now.
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto VPLSpawnPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Respawn", &info->respawn);
  ImGui::DragInt("Max Depth", reinterpret_cast<int*>(&info->maxDepth), 1, 1);
}

auto VPLSpawnPass::execute(RDG::RenderContext* context,
                           RDG::RenderData const& renderData) noexcept -> void {
  // Bind common for RT
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  DVPLPack dvpl_pack = DVPLPack::fetchPack(renderData);
  dvpl_pack.bindPack(this, context);

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  const Math::bounds3 aabb = *(Math::bounds3*)renderData.getPtr("SceneAABB");
  const Math::vec3 center = 0.5 * (aabb.pMax + aabb.pMin);
  const float radius = 0.5 * Math::length(aabb.pMax - aabb.pMin);

  struct PushConstant {
    Math::vec4 boundSphere;
    uint32_t sample_batch;
    int max_depth;
  } pConst = {Math::vec4{center, radius}, renderData.getUInt("AccumIdx"),
              info->maxDepth};

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  if (info->respawn) encoder->traceRays(1, info->maxNumber, 1);
  encoder->end();
}

VPLVisualizePass::VPLVisualizePass(VPLSpawnInfo* info) : info(info) {
  auto [vert, frag] = GFX::ShaderLoader_SLANG::load<2u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vpl/vpl-visualize.slang",
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

  reflector.addInput("VPLPositions")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT));

  reflector.addInput("VPLNormals")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT));

  reflector.addInput("VPLColors")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

  reflector.addInput("CounterBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT));

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

  return reflector;
}

auto VPLVisualizePass::execute(
    RDG::RenderContext* context,
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

  DVPLPack dvpl_pack = DVPLPack::fetchPack(renderData);
  dvpl_pack.bindPack(this, context);

  updateBinding(context, "CameraBuffer",
                renderData.getBindingResource("GlobalUniforms").value());

  RHI::RenderPassEncoder* encoder = beginPass(context, color);
  encoder->pushConstants(&pConst,
                         (uint32_t)RHI::ShaderStages::VERTEX |
                             (uint32_t)RHI::ShaderStages::FRAGMENT,
                         0, sizeof(PushConstantBuffer));

  encoder->draw(30, info->maxNumber, 0, 0);
  encoder->end();
}

auto VPLVisualizePass::renderUI() noexcept -> void {
  ImGui::DragFloat("Size Scale", &pConst.gVPLRenderScale, 0.01f, 0);
  ImGui::DragFloat("Color Scale", &pConst.gVPLColorScale, 0.1f, 0);
}

VPLTestGraph::VPLTestGraph() {
  spawn_info.maxDepth = 2;
  spawn_info.maxNumber = 512 * 256;

  // counter invalid pass
  addPass(std::make_unique<Addon::VPL::CounterInvalidPass>(&spawn_info),
          "CounterInvalid Pass");
  // spawn pass
  addPass(std::make_unique<Addon::VPL::VPLSpawnPass>(&spawn_info),
          "VPLSpawn Pass");
  addEdge("CounterInvalid Pass", "CounterBuffer",
          "VPLSpawn Pass", "CounterBuffer");
  // visualize pass
  addPass(std::make_unique<Addon::VPL::VPLVisualizePass>(&spawn_info),
          "VPLVisualize Pass");
  DVPLPack::addEdge("VPLSpawn Pass", "VPLVisualize Pass", this);

  markOutput("VPLVisualize Pass", "Color");
}
}  // namespace SIByL::Addon::VPL