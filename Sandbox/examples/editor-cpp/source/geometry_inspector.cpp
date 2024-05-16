#include "geometry_inspector.hpp"
#include <seditor-base.hpp>

namespace se {
GeometryInspectorPass::GeometryInspectorPass() {
  auto [vert, frag] = gfx::GFXContext::load_shader_slang(
    "../shaders/editor/mesh-inspector.slang",
    std::array<std::pair<std::string, rhi::ShaderStageBit>, 2>{
      std::make_pair("vertexMain", rhi::ShaderStageBit::VERTEX),
      std::make_pair("fragmentMain", rhi::ShaderStageBit::FRAGMENT),
    });
  rdg::RenderPass::init(vert.get(), frag.get());
  //geo_vis_buffer = GFX::GFXManager::get()->createStructuredUniformBuffer<GeoVisUniform>();
  //std::string const& engine_path = se::RuntimeConfig::get()->string_property("engine_path");
  //matcapGuid = GFX::GFXManager::get()->registerTextureResource(
  //    (engine_path + "/Binaries/Runtime/textures/matcap.png").c_str());

  color = gfx::GFXContext::create_texture_desc(se::rhi::TextureDescriptor{
    {512,512,1}, 1, 1, 1, se::rhi::TextureDimension::TEX2D,
    se::rhi::TextureFormat::RGBA32_FLOAT,
    (uint32_t)se::rhi::TextureUsageBit::COLOR_ATTACHMENT 
    | (uint32_t)se::rhi::TextureUsageBit::TEXTURE_BINDING,
  });
  depth = gfx::GFXContext::create_texture_desc(se::rhi::TextureDescriptor{
    {512,512,1}, 1, 1, 1, se::rhi::TextureDimension::TEX2D,
    se::rhi::TextureFormat::DEPTH32_FLOAT,
    (uint32_t)se::rhi::TextureUsageBit::DEPTH_ATTACHMENT 
    | (uint32_t)se::rhi::TextureUsageBit::TEXTURE_BINDING,
  });
}

auto GeometryInspectorPass::reflect() noexcept -> rdg::PassReflection {
  rdg::PassReflection reflector;
  reflector.addOutput("Color")
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
        .setDepthCompareFn(rhi::CompareFunction::LESS));
  return reflector;
}
//
//auto GeometryInspectorPass::renderUI() noexcept -> void {
//  {  // Select an item type
//    const char* item_names[] = {"BaseColor",
//                                "Metalness",
//                                "Anisotropy",
//                                "Roughness",
//                                "FlatNormal",
//                                "GeometryNormal",
//                                "ShadingNormal",
//                                "NormalMap",
//                                "Opacity",
//                                "Emission",
//                                "SpecularF0",
//                                "VertexColor",
//                                "Matcap",
//                                "MatcapSurface",
//                                "VertexNormal",
//                                "UVChecker"};
//    int debug_mode = int(geo_vis.showEnum);
//    ImGui::Combo("Mode", &debug_mode, item_names, IM_ARRAYSIZE(item_names),
//                 IM_ARRAYSIZE(item_names));
//    geo_vis.showEnum = ShowEnum(debug_mode);
//  }
//
//  // wireframe
//  bool use_wireframe = geo_vis.use_wireframe == 1;
//  ImGui::Checkbox("Use Wireframe", &use_wireframe);
//  geo_vis.use_wireframe = use_wireframe ? 1 : 0;
//  if (geo_vis.use_wireframe) {
//    ImGui::ColorPicker3("Wireframe Color", (float*)&geo_vis.wireframe_color);
//    ImGui::DragFloat("- Wireframe Smoothing X", &geo_vis.wireframe_smoothing, 0.01);
//    ImGui::DragFloat("- Wireframe Thickness", &geo_vis.wireframe_thickness, 0.01);    
//  }
//  ImGui::DragFloat("- CustomData-0", &geo_vis.padding_0, 0.01);    
//  ImGui::DragFloat("- CustomData-1", &geo_vis.padding_1, 0.01);    
//
//}

auto GeometryInspectorPass::execute(
  rdg::RenderContext* context,
  rdg::RenderData const& renderData) noexcept -> void {

  renderPassDescriptor = {
    {rhi::RenderPassColorAttachment{color->getRTV(0, 0, 1),
      nullptr, {0, 0, 0, 1}, rhi::LoadOp::CLEAR, rhi::StoreOp::STORE}},
      rhi::RenderPassDepthStencilAttachment{
        depth->getDSV(0, 0, 1), 1, rhi::LoadOp::CLEAR, rhi::StoreOp::STORE, false,
        0, rhi::LoadOp::LOAD, rhi::StoreOp::DONT_CARE, false},
  };

  //std::vector<rhi::BindGroupEntry>* set_0_entries =
  //  renderData.getBindGroupEntries("CommonScene");

  updateBindings(context, {
    {"GPUScene_camera", scene->getGPUScene()->bindingResourceCamera() } ,
    {"GPUScene_position", scene->getGPUScene()->bindingResourcePosition() },
    {"GPUScene_index", scene->getGPUScene()->bindingResourceIndex() },
    {"GPUScene_vertex", scene->getGPUScene()->bindingResourceVertex() },
    {"GPUScene_geometry", scene->getGPUScene()->bindingResourceGeometry() },
  });

  //GFX::GFXManager::get()->
  //gfx::Texture* matcap_tex =
  //    Core::ResourceManager::get()->getResource<GFX::Texture>(matcapGuid);
  //RHI::Sampler* default_sampler = GFX::GFXManager::get()->samplerTable.fetch(
  //    RHI::AddressMode::CLAMP_TO_EDGE, RHI::FilterMode::LINEAR,
  //    RHI::MipmapFilterMode::LINEAR);

  //geo_vis_buffer.setStructure(geo_vis, context->flightIdx);
  //std::vector<RHI::BindGroupEntry> set_1_entries =
  //    std::vector<RHI::BindGroupEntry>{
  //        RHI::BindGroupEntry{
  //            0, geo_vis_buffer.getBufferBinding(context->flightIdx)},
  //        RHI::BindGroupEntry{
  //            1, RHI::BindingResource{matcap_tex->getSRV(0, 1, 0, 1),
  //                                    default_sampler}}};

  rhi::RenderPassEncoder* encoder = beginPass(context, color.get());

  //renderData.getDelegate("IssueVisibleDrawcalls")(
  //    prepareDelegateData(context, renderData));

  //if (sceneDataPack.geometry_buffer.buffer_host.size() > 0) {
  //    data.passEncoder.render->setIndexBuffer(
  //        sceneDataPack.index_buffer.get_primal(), RHI::IndexFormat::UINT32_T, 0,
  //        sceneDataPack.index_buffer.get_primal()->size());
  //    data.passEncoder.render->setBindGroup(
  //        0, data.pipelinePass->bindgroups[0][flightIdx].get(), 0, 0);

  //    for (auto& pair : sceneDataPack.mesh_ref_record) {
  //        for (auto geometry_idx : pair.second.geometry_indices[pair.second.lod_shown]) {
  //            auto& geometry = sceneDataPack.geometry_buffer.buffer_host[geometry_idx];
  //            data.passEncoder.render->pushConstants(
  //                &geometry_idx, (uint32_t)rhi::ShaderStageBit::VERTEX, 0,
  //                sizeof(uint32_t));
  //            data.passEncoder.render->drawIndexed(
  //                geometry.indexSize, 1, geometry.indexOffset,
  //                geometry.vertexOffset, 0);
  //        }
  //    }
  //}

  struct PushConst {
    int geometryIndex;
    int cameraIndex;
  } pConst;
  pConst.cameraIndex = se::editor::EditorBase::viewportWidget.camera_index;

  encoder->setIndexBuffer(
    scene->getGPUScene()->index_buffer->buffer.get(), rhi::IndexFormat::UINT32_T, 0,
    scene->getGPUScene()->index_buffer->buffer->size());

  std::span<gfx::Scene::GeometryDrawData> geometries =
    scene->gpuScene.geometry_buffer->getHostAsStructuredArray<gfx::Scene::GeometryDrawData>();
  for (size_t geometry_idx = 0; geometry_idx < geometries.size(); ++geometry_idx) {
    auto& geometry = geometries[geometry_idx];
    pConst.geometryIndex = geometry_idx;
    encoder->pushConstants(&pConst, (uint32_t)rhi::ShaderStageBit::VERTEX| (uint32_t)rhi::ShaderStageBit::FRAGMENT, 0, sizeof(PushConst));
    encoder->drawIndexed(
      geometry.indexSize, 1, geometry.indexOffset,
      geometry.vertexOffset, 0);
  }

  encoder->end();
}

GeometryInspectorGraph::GeometryInspectorGraph() {
  addPass(std::make_unique<GeometryInspectorPass>(), "Inspector Pass");
  markOutput("Inspector Pass", "Color");
}
}