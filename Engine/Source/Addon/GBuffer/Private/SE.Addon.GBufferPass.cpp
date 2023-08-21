#include "../Public/SE.Addon.GBufferPass.hpp"
#include <Passes/FullScreenPasses/SE.SRenderer-Blit.hpp>

namespace SIByL::Addon {
RasterizedGBufferPass::RasterizedGBufferPass() {
  auto [vert, frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/gbuffer/"
      "rasterized-gbuffer.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 2>{
          std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RDG::RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto RasterizedGBufferPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addOutput("ViewDepth")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(0));

  reflector.addOutput("DiffuseAlbedo")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(1));

  reflector.addOutput("SpecularRough")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(2));

  reflector.addOutput("Normal")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(3));

  reflector.addOutput("GeometryNormal")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(4));

  reflector.addOutput("Emissive")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(5));

  reflector.addOutput("MotionVector")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(6));

  reflector.addInputOutput("Depth")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::DepthStencilAttachment}
                   .enableDepthWrite(false)
                   .setAttachmentLoc(0)
                   .setDepthCompareFn(RHI::CompareFunction::EQUAL));

  return reflector;
}

auto RasterizedGBufferPass::execute(RDG::RenderContext* context,
                                    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* viewDepth = renderData.getTexture("ViewDepth");
  GFX::Texture* diffuseAlbedo = renderData.getTexture("DiffuseAlbedo");
  GFX::Texture* specularRough = renderData.getTexture("SpecularRough");
  GFX::Texture* normal = renderData.getTexture("Normal");
  GFX::Texture* geometryNormal = renderData.getTexture("GeometryNormal");
  GFX::Texture* emissive = renderData.getTexture("Emissive");
  GFX::Texture* motionVector = renderData.getTexture("MotionVector");
  GFX::Texture* depth = renderData.getTexture("Depth");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{viewDepth->getRTV(0, 0, 1),
                                      nullptr,
                                      {65504.f, 65504.f, 65504.f, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE},
       RHI::RenderPassColorAttachment{diffuseAlbedo->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE},
       RHI::RenderPassColorAttachment{specularRough->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE},
       RHI::RenderPassColorAttachment{normal->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE},
       RHI::RenderPassColorAttachment{geometryNormal->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE},
       RHI::RenderPassColorAttachment{emissive->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE},
       RHI::RenderPassColorAttachment{motionVector->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{
          depth->getDSV(0, 0, 1), 1, RHI::LoadOp::LOAD, RHI::StoreOp::DONT_CARE,
          false, 0, RHI::LoadOp::LOAD, RHI::StoreOp::DONT_CARE, false},
  };

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);

  updateBindings(
      context,
      std::vector<std::pair<std::string, RHI::BindingResource>>{
          std::make_pair(
              "PrevGlobalUniforms",
              renderData.getBindingResource("PrevGlobalUniforms").value()),
          std::make_pair(
              "prevGeometries",
              renderData.getBindingResource("PrevGeometryBuffer").value()),
      });

  RHI::RenderPassEncoder* encoder = beginPass(context, viewDepth);

  renderData.getDelegate("IssueAllDrawcalls")(
      prepareDelegateData(context, renderData));

  encoder->end();
}

enum struct ShowEnum : uint32_t {
  DiffuseAlbedo,
  GeometryNormal,
  ShadingNormal,
  MotionVector,
};

auto GBufferUtils::addGBufferInput(RDG::PassReflection& reflector,
                                   uint32_t stages) noexcept
    -> void {
  reflector.addInput("ViewDepth")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));

  reflector.addInput("DiffuseAlbedo")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));

  reflector.addInput("SpecularRough")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));

  reflector.addInput("Normal")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));

  reflector.addInput("GeometryNormal")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));

  reflector.addInput("Emissive")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));

  reflector.addInput("MotionVector")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));
}

auto GBufferUtils::addPrevGBufferInput(RDG::PassReflection& reflector,
                                       uint32_t stages) noexcept
    -> void {
  reflector.addInput("PrevViewDepth")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));

  reflector.addInput("PrevDiffuseAlbedo")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));

  reflector.addInput("PrevSpecularRough")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));

  reflector.addInput("PrevNormal")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));

  reflector.addInput("PrevGeometryNormal")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));
}

auto GBufferUtils::addPrevGbufferInputOutput(RDG::PassReflection& reflector,
                                       uint32_t stages) noexcept -> void {
  reflector.addInputOutput("PrevViewDepth")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));

  reflector.addInputOutput("PrevDiffuseAlbedo")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));

  reflector.addInputOutput("PrevSpecularRough")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));

  reflector.addInputOutput("PrevNormal")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));

  reflector.addInputOutput("PrevGeometryNormal")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .setSubresource(0, 1, 0, 1)
                   .addStage(stages));
}

auto GBufferUtils::addGBufferEdges(RDG::Graph* graph, std::string const& src,
                                   std::string const& dst) noexcept -> void {
  graph->addEdge(src, "ViewDepth", dst, "ViewDepth");
  graph->addEdge(src, "DiffuseAlbedo", dst, "DiffuseAlbedo");
  graph->addEdge(src, "SpecularRough", dst, "SpecularRough");
  graph->addEdge(src, "Normal", dst, "Normal");
  graph->addEdge(src, "GeometryNormal", dst, "GeometryNormal");
  graph->addEdge(src, "Emissive", dst, "Emissive");
  graph->addEdge(src, "MotionVector", dst, "MotionVector");
}

auto GBufferUtils::addPrevGBufferEdges(RDG::Graph* graph,
                                       std::string const& src,
                                       std::string const& dst) noexcept
    -> void {
  graph->addEdge(src, "ViewDepth", dst, "PrevViewDepth");
  graph->addEdge(src, "DiffuseAlbedo", dst, "PrevDiffuseAlbedo");
  graph->addEdge(src, "SpecularRough", dst, "PrevSpecularRough");
  graph->addEdge(src, "Normal", dst, "PrevNormal");
  graph->addEdge(src, "GeometryNormal", dst, "PrevGeometryNormal");
}

auto GBufferUtils::addBlitPrevGBufferEdges(RDG::Graph* graph,
                                           std::string const& src,
                                           std::string const& tgt,
                                           std::string const& dst) noexcept
    -> void {
  graph->addEdge(src, "ViewDepth", dst, "ViewDepth Source");
  graph->addEdge(src, "DiffuseAlbedo", dst, "DiffuseAlbedo Source");
  graph->addEdge(src, "SpecularRough", dst, "SpecularRough Source");
  graph->addEdge(src, "Normal", dst, "Normal Source");
  graph->addEdge(src, "GeometryNormal", dst, "GeometryNormal Source");

  graph->addEdge(tgt, "PrevViewDepth", dst, "ViewDepth Target");
  graph->addEdge(tgt, "PrevDiffuseAlbedo", dst, "DiffuseAlbedo Target");
  graph->addEdge(tgt, "PrevSpecularRough", dst, "SpecularRough Target");
  graph->addEdge(tgt, "PrevNormal", dst, "Normal Target");
  graph->addEdge(tgt, "PrevGeometryNormal", dst, "GeometryNormal Target");
}

auto GBufferUtils::bindGBufferResource(
    RDG::PipelinePass* pipeline,
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* viewDepth = renderData.getTexture("ViewDepth");
  GFX::Texture* diffuseAlbedo = renderData.getTexture("DiffuseAlbedo");
  GFX::Texture* specularRough = renderData.getTexture("SpecularRough");
  GFX::Texture* normal = renderData.getTexture("Normal");
  GFX::Texture* geometryNormal = renderData.getTexture("GeometryNormal");
  GFX::Texture* emissive = renderData.getTexture("Emissive");
  GFX::Texture* motionVector = renderData.getTexture("MotionVector");

  pipeline->updateBindings(
      context,
      std::vector<std::pair<std::string, RHI::BindingResource>>{
          std::make_pair("t_GBufferDepth",
                         RHI::BindingResource{viewDepth->getSRV(0, 1, 0, 1)}),
          std::make_pair("t_GBufferNormals",
                         RHI::BindingResource{normal->getSRV(0, 1, 0, 1)}),
          std::make_pair(
              "t_GBufferGeoNormals",
              RHI::BindingResource{geometryNormal->getSRV(0, 1, 0, 1)}),
          std::make_pair(
              "t_GBufferDiffuseAlbedo",
              RHI::BindingResource{diffuseAlbedo->getSRV(0, 1, 0, 1)}),
          std::make_pair(
              "t_GBufferSpecularRough",
              RHI::BindingResource{specularRough->getSRV(0, 1, 0, 1)}),
          std::make_pair(
              "t_MotionVectors",
              RHI::BindingResource{motionVector->getSRV(0, 1, 0, 1)}),
      });
}

auto GBufferUtils::bindPrevGBufferResource(
    RDG::PipelinePass* pipeline,
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* prev_viewDepth = renderData.getTexture("PrevViewDepth");
  GFX::Texture* prev_diffuseAlbedo = renderData.getTexture("PrevDiffuseAlbedo");
  GFX::Texture* prev_specularRough = renderData.getTexture("PrevSpecularRough");
  GFX::Texture* prev_normal = renderData.getTexture("PrevNormal");
  GFX::Texture* prev_geometryNormal = renderData.getTexture("PrevGeometryNormal");

  pipeline->updateBindings(
      context,
      std::vector<std::pair<std::string, RHI::BindingResource>>{
          std::make_pair(
              "t_PrevGBufferDepth",
              RHI::BindingResource{prev_viewDepth->getSRV(0, 1, 0, 1)}),
          std::make_pair(
              "t_PrevGBufferNormals",
                         RHI::BindingResource{prev_normal->getSRV(0, 1, 0, 1)}),
          std::make_pair(
              "t_PrevGBufferGeoNormals",
              RHI::BindingResource{prev_geometryNormal->getSRV(0, 1, 0, 1)}),
          std::make_pair("t_PrevGBufferDiffuseAlbedo",
              RHI::BindingResource{prev_diffuseAlbedo->getSRV(0, 1, 0, 1)}),
          std::make_pair(
              "t_PrevGBufferSpecularRough",
              RHI::BindingResource{prev_specularRough->getSRV(0, 1, 0, 1)}),
      });
}

GBufferInspectorPass::GBufferInspectorPass() {
  auto [frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/gbuffer/"
      "gbuffer-inspector.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RDG::FullScreenPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto GBufferInspectorPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  GBufferUtils::addGBufferInput(
      reflector, (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT);

  reflector.addOutput("Output")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .enableDepthWrite(false)
                   .setAttachmentLoc(0)
                   .setDepthCompareFn(RHI::CompareFunction::ALWAYS));

  return reflector;
}

auto GBufferInspectorPass::renderUI() noexcept -> void {
  const char* item_names[] = {
      "DiffuseAlbedo",
      "GeometryNormal",
      "ShadingNormal",
      "MotionVector",
  };
  int debug_mode = int(showEnum);
  ImGui::Combo("Mode", &debug_mode, item_names, IM_ARRAYSIZE(item_names),
               IM_ARRAYSIZE(item_names));
  showEnum = uint32_t(debug_mode);
}

auto GBufferInspectorPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* output = renderData.getTexture("Output");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{output->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{},
  };

  GBufferUtils::bindGBufferResource(this, context, renderData);

  updateBindings(
      context, std::vector<std::pair<std::string, RHI::BindingResource>>{
                   std::make_pair(
                       "CameraDataUniforms",
                       renderData.getBindingResource("GlobalUniforms").value()),
               });

  RHI::RenderPassEncoder* encoder = beginPass(context, output);

  encoder->pushConstants(&showEnum, (uint32_t)RHI::ShaderStages::FRAGMENT, 0,
                         sizeof(uint32_t));
  dispatchFullScreen(context);

  encoder->end();
}

enum struct TemporalShowEnum : uint32_t {
  DiffuseAlbedo,
  GeometryNormal,
  ShadingNormal,
  MotionVector,
  TemporalDiffuseAlbedo,
  AlbedoDiff,
};

GBufferTemporalInspectorPass::GBufferTemporalInspectorPass() {
  auto [frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/gbuffer/"
      "gbuffer-temporal-inspector.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RDG::FullScreenPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto GBufferTemporalInspectorPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  GBufferUtils::addGBufferInput(
      reflector, (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT);
  GBufferUtils::addPrevGbufferInputOutput(
      reflector, (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT);

  reflector.addOutput("Output")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .enableDepthWrite(false)
                   .setAttachmentLoc(0)
                   .setDepthCompareFn(RHI::CompareFunction::ALWAYS));

  return reflector;
}

auto GBufferTemporalInspectorPass::renderUI() noexcept -> void {
  const char* item_names[] = {
      "DiffuseAlbedo",
      "GeometryNormal",
      "ShadingNormal",
      "MotionVector",
      "TemporalDiffuseAlbedo",
      "AlbedoDiff",
  };
  int debug_mode = int(showEnum);
  ImGui::Combo("Mode", &debug_mode, item_names, IM_ARRAYSIZE(item_names),
               IM_ARRAYSIZE(item_names));
  showEnum = uint32_t(debug_mode);
}

auto GBufferTemporalInspectorPass::execute(
    RDG::RenderContext* context,
                                   RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* output = renderData.getTexture("Output");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{output->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{},
  };

  GBufferUtils::bindGBufferResource(this, context, renderData);
  GBufferUtils::bindPrevGBufferResource(this, context, renderData);

  updateBindings(
      context, std::vector<std::pair<std::string, RHI::BindingResource>>{
                   std::make_pair(
                       "CameraDataUniforms",
                       renderData.getBindingResource("GlobalUniforms").value()),
                   std::make_pair(
                       "PrevCameraDataUniforms",
                       renderData.getBindingResource("PrevGlobalUniforms").value()),
               });

  RHI::RenderPassEncoder* encoder = beginPass(context, output);

  encoder->pushConstants(&showEnum, (uint32_t)RHI::ShaderStages::FRAGMENT, 0,
                         sizeof(uint32_t));
  dispatchFullScreen(context);

  encoder->end();
}

GBufferHolderSource::GBufferHolderSource() { RDG::DummyPass::init(); }

auto GBufferHolderSource::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addOutput("ViewDepth")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT);

  reflector.addOutput("DiffuseAlbedo")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT);

  reflector.addOutput("SpecularRough")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT);

  reflector.addOutput("Normal")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT);

  reflector.addOutput("GeometryNormal")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::R32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT);

  return reflector;
}

auto GBufferHolderGraph::alias() noexcept -> RDG::AliasDict {
  RDG::AliasDict dict;
  dict.addAlias("ViewDepth Source", CONCAT("Blit ViewDepth"), "Source");
  dict.addAlias("DiffuseAlbedo Source", CONCAT("Blit DiffuseAlbedo"), "Source");
  dict.addAlias("SpecularRough Source", CONCAT("Blit SpecularRough"), "Source");
  dict.addAlias("Normal Source", CONCAT("Blit Normal"), "Source");
  dict.addAlias("GeometryNormal Source", CONCAT("Blit GeometryNormal"), "Source");

  dict.addAlias("ViewDepth Target", CONCAT("Blit ViewDepth"), "Target");
  dict.addAlias("DiffuseAlbedo Target", CONCAT("Blit DiffuseAlbedo"), "Target");
  dict.addAlias("SpecularRough Target", CONCAT("Blit SpecularRough"), "Target");
  dict.addAlias("Normal Target", CONCAT("Blit Normal"), "Target");
  dict.addAlias("GeometryNormal Target", CONCAT("Blit GeometryNormal"),
                "Target");

  return dict;
}

auto GBufferHolderGraph::onRegister(RDG::Graph* graph) noexcept -> void {
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