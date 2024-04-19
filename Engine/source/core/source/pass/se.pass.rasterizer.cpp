#define DLIB_EXPORT
#include "passes/se.pass.rasterizer.hpp"
#undef DLIB_EXPORT

namespace se {
RasterizerShadePass::RasterizerShadePass() {
  auto [vert, frag] = gfx::GFXContext::load_shader_slang(
    "rasterizer/shade.slang",
    std::array<std::pair<std::string, rhi::ShaderStageBit>, 2>{
      std::make_pair("vertexMain", rhi::ShaderStageBit::VERTEX),
      std::make_pair("fragmentMain", rhi::ShaderStageBit::FRAGMENT),
    });
  rdg::RenderPass::init(vert.get(), frag.get());
}

auto RasterizerShadePass::reflect() noexcept -> rdg::PassReflection {
  rdg::PassReflection reflector;
  reflector.addOutput("Color")
    .isTexture().withSize(se::vec3(1, 1, 1))
    .withFormat(rhi::TextureFormat::RGBA32_FLOAT)
    .withUsages((uint32_t)rhi::TextureUsageBit::COLOR_ATTACHMENT)
    .consume(rdg::TextureInfo::ConsumeEntry{
      rdg::TextureInfo::ConsumeType::ColorAttachment}
        .setAttachmentLoc(0));
  reflector.addOutput("Index")
    .isTexture().withSize(se::vec3(1, 1, 1))
    .withFormat(rhi::TextureFormat::R32_UINT)
    .withUsages((uint32_t)rhi::TextureUsageBit::COLOR_ATTACHMENT)
    .consume(rdg::TextureInfo::ConsumeEntry{
      rdg::TextureInfo::ConsumeType::ColorAttachment}
        .setAttachmentLoc(1));
  reflector.addOutput("Depth")
    .isTexture().withSize(se::vec3(1, 1, 1))
    .withFormat(rhi::TextureFormat::DEPTH32_FLOAT)
    .withUsages((uint32_t)rhi::TextureUsageBit::DEPTH_ATTACHMENT)
    .consume(rdg::TextureInfo::ConsumeEntry{
      rdg::TextureInfo::ConsumeType::DepthStencilAttachment}
        .enableDepthWrite(true)
        .setAttachmentLoc(0)
        .setDepthCompareFn(rhi::CompareFunction::LESS_EQUAL));
  return reflector;
}

auto RasterizerShadePass::execute(
  rdg::RenderContext* context,
  rdg::RenderData const& renderData) noexcept -> void {
  gfx::TextureHandle color = renderData.getTexture("Color");
  gfx::TextureHandle index = renderData.getTexture("Index");
  gfx::TextureHandle depth = renderData.getTexture("Depth");
  gfx::SceneHandle scene = renderData.getScene();

  setRenderPassDescriptor(rhi::RenderPassDescriptor{
    { rhi::RenderPassColorAttachment{color->getRTV(0, 0, 1),
      nullptr, {0, 0, 0, 1}, rhi::LoadOp::CLEAR, rhi::StoreOp::STORE},
      rhi::RenderPassColorAttachment{index->getRTV(0, 0, 1),
      nullptr, {0, 0, 0, 1}, rhi::LoadOp::CLEAR, rhi::StoreOp::STORE},},
      rhi::RenderPassDepthStencilAttachment{
        depth->getDSV(0, 0, 1), 1, rhi::LoadOp::CLEAR, rhi::StoreOp::STORE, false,
        0, rhi::LoadOp::DONT_CARE, rhi::StoreOp::DONT_CARE, false},
      });
  
  updateBindings(context, {
    {"GPUScene_camera", scene->getGPUScene()->bindingResourceCamera() } ,
    {"GPUScene_position", scene->getGPUScene()->bindingResourcePosition() },
    {"GPUScene_index", scene->getGPUScene()->bindingResourceIndex() },
    {"GPUScene_vertex", scene->getGPUScene()->bindingResourceVertex() },
    {"GPUScene_geometry", scene->getGPUScene()->bindingResourceGeometry() },
  });

  pConst.resolution = { color->texture->width(), color->texture->width() };
  pConst.cameraIndex = scene->getEditorActiveCameraIndex();

  rhi::RenderPassEncoder* encoder = beginPass(context, color.get());
  issueDirectDrawcalls(encoder, scene);
  encoder->end();
}

auto RasterizerShadePass::beforeDirectDrawcall(rhi::RenderPassEncoder* encoder,
  int geometry_idx, gfx::Scene::GeometryDrawData const& geometry) noexcept -> void {
  pConst.geometryIndex = geometry_idx;
  encoder->pushConstants(&pConst, (uint32_t)rhi::ShaderStageBit::VERTEX | (uint32_t)rhi::ShaderStageBit::FRAGMENT, 0, sizeof(PushConst));
}
}