#define DLIB_EXPORT
#include <passes/se.pass.editor.hpp>
#undef DLIB_EXPORT
#include <imgui.h>

namespace se {
EditorInitPass::EditorInitPass() {
  auto [frag] = gfx::GFXContext::load_shader_slang(
    "editor/initialize.slang",
    std::array<std::pair<std::string, rhi::ShaderStageBit>, 1>{
      std::make_pair("fragmentMain", rhi::ShaderStageBit::FRAGMENT),
    });
  rdg::FullScreenPass::init(frag.get());
}

auto EditorInitPass::reflect() noexcept -> rdg::PassReflection {
rdg::PassReflection reflector;
  reflector.addOutput("Index")
    .isTexture().withSize(se::vec3(1, 1, 1))
    .withFormat(rhi::TextureFormat::R32_UINT)
    .withUsages((uint32_t)rhi::TextureUsageBit::COLOR_ATTACHMENT)
    .consume(rdg::TextureInfo::ConsumeEntry{
      rdg::TextureInfo::ConsumeType::ColorAttachment}
        .setAttachmentLoc(0));
  reflector.addOutput("Depth")
    .isTexture().withSize(se::vec3(1, 1, 1))
    .withFormat(rhi::TextureFormat::DEPTH32_FLOAT)
    .withUsages((uint32_t)rhi::TextureUsageBit::DEPTH_ATTACHMENT)
    .consume(rdg::TextureInfo::ConsumeEntry{
      rdg::TextureInfo::ConsumeType::DepthStencilAttachment}
        .enableDepthWrite(true)
        .setAttachmentLoc(0)
        .setDepthCompareFn(rhi::CompareFunction::ALWAYS));
  return reflector;
}

auto EditorInitPass::execute(
  rdg::RenderContext* context,
  rdg::RenderData const& renderData) noexcept -> void {
  gfx::TextureHandle index = renderData.getTexture("Index");
  gfx::TextureHandle depth = renderData.getTexture("Depth");

  renderPassDescriptor = {
    {rhi::RenderPassColorAttachment{index->getRTV(0, 0, 1),
      nullptr, {0, 0, 0, 1}, rhi::LoadOp::CLEAR, rhi::StoreOp::STORE},},
      rhi::RenderPassDepthStencilAttachment{
        depth->getDSV(0, 0, 1), 1, rhi::LoadOp::CLEAR, rhi::StoreOp::STORE, false,
        0, rhi::LoadOp::LOAD, rhi::StoreOp::DONT_CARE, false},
  };

  rhi::RenderPassEncoder* encoder = beginPass(context, index.get());
  dispatchFullScreen(context);
  encoder->end();
}

BillboardPass::BillboardPass() {
  auto [vert, frag] = gfx::GFXContext::load_shader_slang(
    "editor/draw-billboard.slang",
    std::array<std::pair<std::string, rhi::ShaderStageBit>, 2>{
      std::make_pair("vertexMain", rhi::ShaderStageBit::VERTEX),
      std::make_pair("fragmentMain", rhi::ShaderStageBit::FRAGMENT),
    });
  rdg::RenderPass::init(vert.get(), frag.get());

  sampler = gfx::GFXContext::create_sampler_desc(
    rhi::AddressMode::CLAMP_TO_EDGE, 
    rhi::FilterMode::LINEAR, 
    rhi::MipmapFilterMode::LINEAR);

  icons = gfx::GFXContext::create_texture_file(
    RuntimeConfig::get()->string_property("engine_path") + 
    "binary/resources/icons/editor-icons.png");
  gfx::GFXContext::device->trainsitionTextureLayout(icons->texture.get(),
      se::rhi::TextureLayout::UNDEFINED, se::rhi::TextureLayout::GENERAL);
  billboards = gfx::GFXContext::load_buffer_empty();
  billboards->usages = (uint32_t)rhi::BufferUsageBit::STORAGE;
}

auto BillboardPass::reflect() noexcept -> rdg::PassReflection {
rdg::PassReflection reflector;
  reflector.addInputOutput("Color")
    .isTexture().withSize(se::vec3(1, 1, 1))
    .withFormat(rhi::TextureFormat::RGBA32_FLOAT)
    .withUsages((uint32_t)rhi::TextureUsageBit::COLOR_ATTACHMENT)
    .consume(rdg::TextureInfo::ConsumeEntry{
      rdg::TextureInfo::ConsumeType::ColorAttachment}
        .setAttachmentLoc(0));
  reflector.addInputOutput("Depth")
    .isTexture().withSize(se::vec3(1, 1, 1))
    .withFormat(rhi::TextureFormat::DEPTH32_FLOAT)
    .withUsages((uint32_t)rhi::TextureUsageBit::DEPTH_ATTACHMENT)
    .consume(rdg::TextureInfo::ConsumeEntry{
      rdg::TextureInfo::ConsumeType::DepthStencilAttachment}
        .enableDepthWrite(true)
        .setAttachmentLoc(0)
        .setDepthCompareFn(rhi::CompareFunction::LESS_EQUAL));
  reflector.addInputOutput("Index")
    .isTexture().withSize(se::vec3(1, 1, 1))
    .withFormat(rhi::TextureFormat::R32_UINT)
    .withUsages((uint32_t)rhi::TextureUsageBit::COLOR_ATTACHMENT)
    .consume(rdg::TextureInfo::ConsumeEntry{
      rdg::TextureInfo::ConsumeType::ColorAttachment}
        .setAttachmentLoc(1));
  return reflector;
}

struct BillboardData {
  se::vec3 position;
  int iconIDX;
  se::vec2 size;
  int recordID;
  int padding;
};

auto BillboardPass::execute(
  rdg::RenderContext* context,
  rdg::RenderData const& renderData) noexcept -> void {
  gfx::TextureHandle color = renderData.getTexture("Color");
  gfx::TextureHandle index = renderData.getTexture("Index");
  gfx::TextureHandle depth = renderData.getTexture("Depth");
  gfx::SceneHandle scene = renderData.getScene();

  rhi::BindingResource billboards_binding; int size;
  if (external_billboards == nullptr) {
    std::vector<BillboardData> billboards_vec;
    std::span<gfx::Scene::CameraData> cameras =
      scene->getGPUScene()->camera_buffer->getHostAsStructuredArray<gfx::Scene::CameraData>();
    billboards_vec.resize(cameras.size());
    for (int i = 0; i < cameras.size(); ++i) {
        billboards_vec[i] = { cameras[i].posW, 0, {1.f,1.f}, i, 0 };
    }
    billboards->host.resize(billboards_vec.size() * sizeof(BillboardData));
    memcpy(billboards->host.data(), billboards_vec.data(), billboards_vec.size() * sizeof(BillboardData));
    billboards->host_stamp++;
    billboards->hostToDevice();
    billboards_binding = rhi::BindingResource{ {billboards->buffer.get(), 0, billboards->buffer->size()} };
    size = billboards_vec.size();
  }
  else {
    billboards_binding = rhi::BindingResource{ {external_billboards, 0, external_billboards->size()} };
    size = external_billboards->size() / sizeof(BillboardData);
  }

  renderPassDescriptor = {
    { rhi::RenderPassColorAttachment{color->getRTV(0, 0, 1),
      nullptr, {0, 0, 0, 1}, rhi::LoadOp::LOAD, rhi::StoreOp::STORE},
      rhi::RenderPassColorAttachment{index->getRTV(0, 0, 1),
      nullptr, {0, 0, 0, 1}, rhi::LoadOp::LOAD, rhi::StoreOp::STORE},},
      rhi::RenderPassDepthStencilAttachment{
        depth->getDSV(0, 0, 1), 1, rhi::LoadOp::LOAD, rhi::StoreOp::STORE, false,
        0, rhi::LoadOp::DONT_CARE, rhi::StoreOp::DONT_CARE, false},
  };

  updateBindings(context, {
    {"GPUScene_camera", scene->getGPUScene()->bindingResourceCamera() } ,
    {"IconsTextures", icons->getUAV(0,0,1) } ,
    {"IconsSampler", sampler.get() } ,
    {"u_billboards", billboards_binding } ,
  });

  rhi::RenderPassEncoder* encoder = beginPass(context, color.get());

  struct PushConst {
    int cameraIndex;
  } pConst;
  pConst.cameraIndex = scene->editorInfo.active_camera_index;

  encoder->pushConstants(&pConst, (uint32_t)rhi::ShaderStageBit::VERTEX, 0, sizeof(PushConst));
  encoder->draw(6, size, 0, 0);
  encoder->end();
}

auto BillboardPass::setExternalBuffer(rhi::Buffer* buffer) noexcept -> void {
  external_billboards = buffer;
}

Line3DPass::Line3DPass() {
  auto [vert, frag] = gfx::GFXContext::load_shader_slang(
    "editor/draw-line3d.slang",
    std::array<std::pair<std::string, rhi::ShaderStageBit>, 2>{
      std::make_pair("vertexMain", rhi::ShaderStageBit::VERTEX),
      std::make_pair("fragmentMain", rhi::ShaderStageBit::FRAGMENT),
    });
  rdg::RenderPass::init(vert.get(), frag.get());

  lines = gfx::GFXContext::load_buffer_empty();
  lines->usages = (uint32_t)rhi::BufferUsageBit::STORAGE;
}

auto Line3DPass::reflect() noexcept -> rdg::PassReflection {
rdg::PassReflection reflector;
  reflector.addInputOutput("Color")
    .isTexture().withSize(se::vec3(1, 1, 1))
    .withFormat(rhi::TextureFormat::RGBA32_FLOAT)
    .withUsages((uint32_t)rhi::TextureUsageBit::COLOR_ATTACHMENT)
    .consume(rdg::TextureInfo::ConsumeEntry{
      rdg::TextureInfo::ConsumeType::ColorAttachment}
        .setAttachmentLoc(0));
  reflector.addInputOutput("Depth")
    .isTexture().withSize(se::vec3(1, 1, 1))
    .withFormat(rhi::TextureFormat::DEPTH32_FLOAT)
    .withUsages((uint32_t)rhi::TextureUsageBit::DEPTH_ATTACHMENT)
    .consume(rdg::TextureInfo::ConsumeEntry{
      rdg::TextureInfo::ConsumeType::DepthStencilAttachment}
        .enableDepthWrite(true)
        .setAttachmentLoc(0)
        .setDepthCompareFn(rhi::CompareFunction::LESS_EQUAL));
  reflector.addInputOutput("Index")
    .isTexture().withSize(se::vec3(1, 1, 1))
    .withFormat(rhi::TextureFormat::R32_UINT)
    .withUsages((uint32_t)rhi::TextureUsageBit::COLOR_ATTACHMENT)
    .consume(rdg::TextureInfo::ConsumeEntry{
      rdg::TextureInfo::ConsumeType::ColorAttachment}
        .setAttachmentLoc(1));
  return reflector;
}

struct Line3DData {
  se::vec3 position_1;
  float a;
  se::vec3 position_2;
  float b;
  se::vec3 color;
  float width;
};

auto Line3DPass::execute(
  rdg::RenderContext* context,
  rdg::RenderData const& renderData) noexcept -> void {
  gfx::TextureHandle color = renderData.getTexture("Color");
  gfx::TextureHandle index = renderData.getTexture("Index");
  gfx::TextureHandle depth = renderData.getTexture("Depth");
  gfx::SceneHandle scene = renderData.getScene();

  //billboards->host_stamp++;
  rhi::BindingResource lines_binding; int size;
  if (external_lines == nullptr) {
    lines->hostToDevice();
    lines_binding = rhi::BindingResource{ {lines->buffer.get(), 0, lines->buffer->size()} };
    auto linevec = lines->getHostAsStructuredArray<Line3DData>();
    size = linevec.size();
  }
  else {
    lines_binding = rhi::BindingResource{ {external_lines, 0, external_lines->size()} };
    size = external_count;
  }

  renderPassDescriptor = {
    { rhi::RenderPassColorAttachment{color->getRTV(0, 0, 1),
      nullptr, {0, 0, 0, 1}, rhi::LoadOp::LOAD, rhi::StoreOp::STORE},
      rhi::RenderPassColorAttachment{index->getRTV(0, 0, 1),
      nullptr, {0, 0, 0, 1}, rhi::LoadOp::LOAD, rhi::StoreOp::STORE},},
      rhi::RenderPassDepthStencilAttachment{
        depth->getDSV(0, 0, 1), 1, rhi::LoadOp::LOAD, rhi::StoreOp::STORE, false,
        0, rhi::LoadOp::DONT_CARE, rhi::StoreOp::DONT_CARE, false},
  };

  updateBindings(context, {
    {"GPUScene_camera", scene->getGPUScene()->bindingResourceCamera() } ,
    {"u_lines", lines_binding } ,
  });

  rhi::RenderPassEncoder* encoder = beginPass(context, color.get());

  struct PushConst {
    se::ivec2 size;
    int cameraIndex;
  } pConst;
  pConst.size = se::ivec2{ color->texture->width() , color->texture->height() };
  pConst.cameraIndex = scene->editorInfo.active_camera_index;

  if (size > 0) {
    encoder->pushConstants(&pConst, (uint32_t)rhi::ShaderStageBit::VERTEX, 0, sizeof(PushConst));
    encoder->draw(30, size, 0, 0);
  }
  encoder->end();
}

auto Line3DPass::clear() noexcept -> void {
  lines->host.resize(0);
  lines->host_stamp++;
}

auto Line3DPass::addLine(se::vec3 a, se::vec3 b, se::vec3 color, float width) noexcept -> void {
  auto line = Line3DData{ a, 0, b, 0, color, width };
  const size_t address = lines->host.size();
  lines->host.resize(address + sizeof(Line3DData));
  memcpy(&lines->host[address], &line, sizeof(Line3DData));
}

auto Line3DPass::addAABB(se::bounds3 aabb, se::vec3 color, float width) noexcept -> void {
  addLine(aabb.corner(0), aabb.corner(1), color, width);
  addLine(aabb.corner(0), aabb.corner(2), color, width);
  addLine(aabb.corner(0), aabb.corner(4), color, width);
  addLine(aabb.corner(7), aabb.corner(3), color, width);
  addLine(aabb.corner(7), aabb.corner(5), color, width);
  addLine(aabb.corner(7), aabb.corner(6), color, width);
  addLine(aabb.corner(4), aabb.corner(6), color, width);
  addLine(aabb.corner(1), aabb.corner(5), color, width);
  addLine(aabb.corner(2), aabb.corner(3), color, width);
  addLine(aabb.corner(2), aabb.corner(6), color, width);
  addLine(aabb.corner(4), aabb.corner(5), color, width);
  addLine(aabb.corner(1), aabb.corner(3), color, width);
}

auto Line3DPass::setExternalBuffer(rhi::Buffer* buffer, size_t line_count) noexcept -> void {
  external_lines = buffer;
  external_count = line_count;
}
}