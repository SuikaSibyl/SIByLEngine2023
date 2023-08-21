#include "../Public/SE.Addon.VXGI.hpp"

namespace SIByL::Addon::VXGI {
VoxelizePass::VoxelizePass(VXGISetting* setting) : setting(setting) {
  auto [vert, geom, frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxgi/"
      "voxelizer.slang",
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
      GFX::GFXManager::get()->createStructuredUniformBuffer<VoxelizeUniform>();
}

auto VoxelizePass::reflect() noexcept -> RDG::PassReflection {
  uint32_t const size = setting->clipmapSetting.size;

  RDG::PassReflection reflector;
  reflector.addOutput("OpacityTex")
      .isTexture()
      .withSize(Math::ivec3(size, size, size))
      .withFormat(RHI::TextureFormat::R16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
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

auto VoxelizePass::execute(RDG::RenderContext* context,
                           RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* depth = renderData.getTexture("Depth");
  GFX::Texture* opacityTex = renderData.getTexture("OpacityTex");

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

  updateBinding(context, "uOpacityVox",
                RHI::BindingResource{opacityTex->getUAV(0, 0, 1)});

  Math::bounds3 aabb =
      *static_cast<Math::bounds3 const*>(renderData.getPtr("SceneAABB"));
  gUniform.aabbMin = aabb.pMin;
  gUniform.aabbMax = aabb.pMax;
  gUniform.voxelSize = setting->clipmapSetting.size;

  uniformBuffer.setStructure(gUniform, context->flightIdx);
  setting->shared.voxUniBinding =
      uniformBuffer.getBufferBinding(context->flightIdx);
  updateBinding(context, "VoxelizerUniform", setting->shared.voxUniBinding);


  //// GFX::GFXManager::get()->
  //GFX::Texture* matcap_tex =
  //    Core::ResourceManager::get()->getResource<GFX::Texture>(matcapGuid);
  //GFX::Sampler* default_sampler =
  //    Core::ResourceManager::get()->getResource<GFX::Sampler>(
  //        GFX::GFXManager::get()->commonSampler.defaultSampler);

  //geo_vis_buffer.setStructure(geo_vis, context->flightIdx);
  //std::vector<RHI::BindGroupEntry> set_1_entries =
  //    std::vector<RHI::BindGroupEntry>{
  //        RHI::BindGroupEntry{
  //            0, geo_vis_buffer.getBufferBinding(context->flightIdx)},
  //        RHI::BindGroupEntry{
  //            1, RHI::BindingResource{matcap_tex->getSRV(0, 1, 0, 1),
  //                                    default_sampler->sampler.get()}}};
  //getBindGroup(context, 1)->updateBinding(set_1_entries);

  RHI::RenderPassEncoder* encoder = beginPass(context, depth);

  renderData.getDelegate("IssueAllDrawcalls")(
      prepareDelegateData(context, renderData));

  encoder->end();
}

VoxelMipPass::VoxelMipPass(VXGISetting* setting)
    : setting(setting) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxgi/"
      "voxel-downsampling.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VoxelMipPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInput("OpacityTex")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

  uint32_t const size = setting->clipmapSetting.size;
  uint32_t const mip1_size = size >> 1;

  reflector.addOutput("OpacityMip")
      .isTexture()
      .withSize(Math::ivec3(mip1_size, mip1_size, mip1_size))
      .withFormat(RHI::TextureFormat::R16_FLOAT)
      .withLevels(setting->clipmapSetting.mip - 1)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT)
                   .setSubresource(0, setting->clipmapSetting.mip - 1, 0, 1));

  return reflector;
}

auto VoxelMipPass::execute(RDG::RenderContext* context,
                                 RDG::RenderData const& renderData) noexcept
    -> void {
  //GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* opacityTex = renderData.getTexture("OpacityTex");
  GFX::Texture* opacityMip = renderData.getTexture("OpacityMip");

  updateBinding(context, "uOpacityVox",
                RHI::BindingResource{opacityTex->getUAV(0, 0, 1)});
  updateBinding(context, "uOpacityVox_mip1",
                RHI::BindingResource{opacityMip->getUAV(0, 0, 1)});
  updateBinding(context, "uOpacityVox_mip2",
                RHI::BindingResource{opacityMip->getUAV(1, 0, 1)});
  updateBinding(context, "uOpacityVox_mip3",
                RHI::BindingResource{opacityMip->getUAV(2, 0, 1)});

  RHI::ComputePassEncoder* encoder = beginPass(context);
  //encoder->pushConstants(&setting->element_count,
  //                       (uint32_t)RHI::ShaderStages::COMPUTE, 0,
  //                       sizeof(uint32_t));
  uint32_t const size = setting->clipmapSetting.size;
  uint32_t const dispatchSize = size / 8;
  if (updateMip) {
    encoder->dispatchWorkgroups(dispatchSize, dispatchSize, dispatchSize);
    //updateMip = false;
  }
  encoder->end();
}

auto VoxelMipPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Update MIP", &updateMip);
}

VoxelVisualizePass::VoxelVisualizePass(VXGISetting* setting)
    : setting(setting) {
  auto [frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/vxgi/"
      "voxel-viewer.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RDG::FullScreenPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto VoxelVisualizePass::reflect() noexcept -> RDG::PassReflection {
  uint32_t const size = setting->clipmapSetting.size;

  RDG::PassReflection reflector;

  reflector.addInput("OpacityTex")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

  reflector.addInput("OpacityMip")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::StorageBinding}
                   .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT)
                   .setSubresource(0, setting->clipmapSetting.mip - 1, 0, 1));

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

auto VoxelVisualizePass::execute(RDG::RenderContext* context,
                                 RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* opacityTex = renderData.getTexture("OpacityTex");
  GFX::Texture* OpacityMip = renderData.getTexture("OpacityMip");

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
  updateBinding(context, "uOpacityVox",
                RHI::BindingResource{opacityTex->getUAV(0, 0, 1)});
  updateBinding(context, "uOpacityMip",
                RHI::BindingResource{OpacityMip->getSRV(0, 3, 0, 1)});


  RHI::RenderPassEncoder* encoder = beginPass(context, color);

  pConst.resolution = Math::ivec2{1280, 720};

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::FRAGMENT, 0,
                         sizeof(PushConstant));

  dispatchFullScreen(context);

  encoder->end();
}

auto VoxelVisualizePass::renderUI() noexcept -> void {
  ImGui::DragFloat("Edge Threshold", &pConst.edgeThreshold, 0.005, 0, 1);
  ImGui::DragInt("Mip Level", &pConst.mipLevelShown, 1, 0,
                 setting->clipmapSetting.mip);
}

struct alignas(64) ConeDebugUniform {
  Math::vec3 position;
  uint32_t geometryID;
  Math::vec3 normal;
  uint32_t validFlags;
  int coneID;
};

ConeTraceDebuggerPass::ConeTraceDebuggerPass(VXGISetting* setting)
    : setting(setting) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "vxgi/conetrace-debugger.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto ConeTraceDebuggerPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInputOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

  reflector.addInternal("DebugUniform")
      .isBuffer()
      .withSize(sizeof(ConeDebugUniform))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

  reflector.addInternal("DebugTex")
      .isTexture()
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withSize(Math::vec3(1, 1, 1))
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
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::StorageBinding}
                .addStage(
                    (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR)
                .setSubresource(0, setting->clipmapSetting.mip, 0, 1));

  return reflector;
}

auto ConeTraceDebuggerPass::renderUI() noexcept -> void {
  ImGui::DragFloat("Cone Length", &coneLength);
  ImGui::DragFloat("Maximum tan", &maximum_tan, 0.01, 0, 1);
  ImGui::DragInt("Debug pixel x", &debugPixel.x);
  ImGui::DragInt("Debug pixel y", &debugPixel.y);
  ImGui::DragInt("Debug voxel x", &debugVoxel.x);
  ImGui::DragInt("Debug voxel y", &debugVoxel.y);
  ImGui::DragInt("Debug voxel z", &debugVoxel.z);
  ImGui::DragInt("Debug voxel w", &debugVoxel.w);

  {  // Select an item type
    const char* item_names[] = {
        "Nearest",
        "Centric",
    };
    int voxel_mode = voxelShown;
    ImGui::Combo("Sample Mode", &voxel_mode, item_names,
                 IM_ARRAYSIZE(item_names), IM_ARRAYSIZE(item_names));
    voxelShown = int(voxel_mode);
  }
  {  // Select an item type
    const char* item_names[] = {
        "Diffuse",
        "Specular",
        "ToVoxel",
    };
    ImGui::Combo("Debug Mode", &debug_mode, item_names,
                 IM_ARRAYSIZE(item_names), IM_ARRAYSIZE(item_names));
  }
}

auto ConeTraceDebuggerPass::execute(RDG::RenderContext* context,
                              RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* debug = renderData.getTexture("DebugTex");
  GFX::Buffer* uniform = renderData.getBuffer("DebugUniform");
  GFX::Texture* radopaTex[6];
  for (int i = 0; i < 6; ++i)
    radopaTex[i] = renderData.getTexture("RadOpaVox6DTex" + std::to_string(i));

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
  updateBinding(context, "u_DebugUniform",
                RHI::BindingResource{
                    {uniform->buffer.get(), 0, uniform->buffer->size()}});
  updateBinding(context, "uRadopaVox6D",
                RHI::BindingResource{std::vector<RHI::TextureView*>{
                    radopaTex[0]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
                    radopaTex[1]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
                    radopaTex[2]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
                    radopaTex[3]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
                    radopaTex[4]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
                    radopaTex[5]->getSRV(0, setting->clipmapSetting.mip, 0, 1),
                }});
  updateBinding(
      context, "uTex3DSampler",
      RHI::BindingResource{setting->shared.tex3dSampler->sampler.get()});
  updateBinding(context, "VoxelizerUniform", setting->shared.voxUniBinding);

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("AccumIdx");

  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
    int invalidSet;
    Math::ivec2 debug_pixel;
    float coneLength;
    int voxelShown;
    Math::ivec4 debugVoxel;
    int debugmode;
    float maximum_tan;
  };
  PushConstant pConst = {color->texture->width(),
                         color->texture->height(),
                         batchIdx,
                         0,
                         debugPixel,
                         coneLength,
                         voxelShown,
                         debugVoxel,
                         debug_mode,
                         maximum_tan};
  if (invalidDebugPixel > 0) {
    pConst.invalidSet = invalidDebugPixel;
    invalidDebugPixel = 0;
  }
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(color->texture->width(), color->texture->height(), 1);

  encoder->end();
}

auto ConeTraceDebuggerPass::onInteraction(
    Platform::Input* input, Editor::Widget::WidgetInfo* info) noexcept -> void {
  if (info->isFocused && info->isHovered) {
    if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
      debugPixel.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
      debugPixel.y = std::clamp(info->mousePos.y, 0.f, 719.f);
      if (input->isKeyPressed(Platform::SIByL_KEY_SPACE)) {
        invalidDebugPixel = 2;
      } else {
        invalidDebugPixel = 1;
      }
    }
  }
}
}  // namespace SIByL::Addon::VXGI