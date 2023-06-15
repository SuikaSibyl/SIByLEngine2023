#include "SE.SRenderer-GeometryInspectorPass.hpp"

namespace SIByL {
GeometryInspectorPass::GeometryInspectorPass() {
  vert = GFX::GFXManager::get()->registerShaderModuleResource(
      "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/"
      "mesh_inspector/mesh_inspector_vert.spv",
      {nullptr, RHI::ShaderStages::VERTEX});
  frag = GFX::GFXManager::get()->registerShaderModuleResource(
      "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/"
      "mesh_inspector/mesh_inspector_frag.spv",
      {nullptr, RHI::ShaderStages::FRAGMENT});
  RDG::RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
  geo_vis_buffer =
      GFX::GFXManager::get()->createStructuredUniformBuffer<GeoVisUniform>();
}

auto GeometryInspectorPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
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
                   .enableDepthWrite(false)
                   .setAttachmentLoc(0)
                   .setDepthCompareFn(RHI::CompareFunction::EQUAL));

  return reflector;
}

auto GeometryInspectorPass::renderUI() noexcept -> void {
  // wireframe
  bool use_wireframe = geo_vis.use_wireframe == 1;
  ImGui::Checkbox("Use Wireframe", &use_wireframe);
  geo_vis.use_wireframe = use_wireframe ? 1 : 0;
  if (geo_vis.use_wireframe) {
    ImGui::DragFloat("- Wireframe Smoothing X", &geo_vis.wireframe_smoothing, 0.01);
    ImGui::DragFloat("- Wireframe Thickness", &geo_vis.wireframe_thickness, 0.01);    
  }
}

auto GeometryInspectorPass::execute(
    RDG::RenderContext* context,
                     RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* depth = renderData.getTexture("Depth");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{color->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{
          depth->getDSV(0, 0, 1), 1, RHI::LoadOp::LOAD, RHI::StoreOp::DONT_CARE,
          false, 0, RHI::LoadOp::LOAD, RHI::StoreOp::DONT_CARE, false},
  };

  RHI::RenderPassEncoder* encoder = beginPass(context, color);

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");

  geo_vis_buffer.setStructure(geo_vis, context->flightIdx);
  std::vector<RHI::BindGroupEntry> set_1_entries =
      std::vector<RHI::BindGroupEntry>{RHI::BindGroupEntry{
          0, geo_vis_buffer.getBufferBinding(context->flightIdx)}};

  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  getBindGroup(context, 1)->updateBinding(set_1_entries);

  renderData.getDelegate("IssueAllDrawcalls")(
      prepareDelegateData(context, renderData));

  encoder->end();
}
}