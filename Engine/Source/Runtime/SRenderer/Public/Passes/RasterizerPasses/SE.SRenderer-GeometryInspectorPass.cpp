#include "SE.SRenderer-GeometryInspectorPass.hpp"
#include <Config/SE.Core.Config.hpp>

namespace SIByL {
GeometryInspectorPass::GeometryInspectorPass() {
  auto [vert, frag] = GFX::ShaderLoader_SLANG::load(
    "../Engine/Shaders/SRenderer/rasterizer/mesh_inspector.slang",
    std::array<std::pair<std::string, RHI::ShaderStages>, 2>{
      std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
      std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
    });
  RDG::RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));

  geo_vis_buffer = GFX::GFXManager::get()->createStructuredUniformBuffer<GeoVisUniform>();
  std::string const& engine_path = Core::RuntimeConfig::get()->string_property("engine_path");
  matcapGuid = GFX::GFXManager::get()->registerTextureResource(
      (engine_path + "/Binaries/Runtime/textures/matcap.png").c_str());
}

auto GeometryInspectorPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("Color")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
            .setAttachmentLoc(0));
  reflector.addInputOutput("Depth")
      .isTexture().withSize(Math::vec3(1, 1, 1))
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
  {  // Select an item type
    const char* item_names[] = {"BaseColor",
                                "Metalness",
                                "Anisotropy",
                                "Roughness",
                                "FlatNormal",
                                "GeometryNormal",
                                "ShadingNormal",
                                "NormalMap",
                                "Opacity",
                                "Emission",
                                "SpecularF0",
                                "VertexColor",
                                "Matcap",
                                "MatcapSurface",
                                "VertexNormal",
                                "UVChecker"};
    int debug_mode = int(geo_vis.showEnum);
    ImGui::Combo("Mode", &debug_mode, item_names, IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
    geo_vis.showEnum = ShowEnum(debug_mode);
  }

  // wireframe
  bool use_wireframe = geo_vis.use_wireframe == 1;
  ImGui::Checkbox("Use Wireframe", &use_wireframe);
  geo_vis.use_wireframe = use_wireframe ? 1 : 0;
  if (geo_vis.use_wireframe) {
    ImGui::ColorPicker3("Wireframe Color", (float*)&geo_vis.wireframe_color);
    ImGui::DragFloat("- Wireframe Smoothing X", &geo_vis.wireframe_smoothing, 0.01);
    ImGui::DragFloat("- Wireframe Thickness", &geo_vis.wireframe_thickness, 0.01);    
  }
  ImGui::DragFloat("- CustomData-0", &geo_vis.padding_0, 0.01);    
  ImGui::DragFloat("- CustomData-1", &geo_vis.padding_1, 0.01);    

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

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");

  //GFX::GFXManager::get()->
  GFX::Texture* matcap_tex =
      Core::ResourceManager::get()->getResource<GFX::Texture>(matcapGuid);
  RHI::Sampler* default_sampler = GFX::GFXManager::get()->samplerTable.fetch(
      RHI::AddressMode::CLAMP_TO_EDGE, RHI::FilterMode::LINEAR,
      RHI::MipmapFilterMode::LINEAR);

  geo_vis_buffer.setStructure(geo_vis, context->flightIdx);
  std::vector<RHI::BindGroupEntry> set_1_entries =
      std::vector<RHI::BindGroupEntry>{
          RHI::BindGroupEntry{
              0, geo_vis_buffer.getBufferBinding(context->flightIdx)},
          RHI::BindGroupEntry{
              1, RHI::BindingResource{matcap_tex->getSRV(0, 1, 0, 1),
                                      default_sampler}}};

  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  getBindGroup(context, 1)->updateBinding(set_1_entries);

  RHI::RenderPassEncoder* encoder = beginPass(context, color);

  renderData.getDelegate("IssueVisibleDrawcalls")(
      prepareDelegateData(context, renderData));

  encoder->end();
}
}