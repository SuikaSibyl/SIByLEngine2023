#include "SE.SRenderer-BarLCPass.hpp"

namespace SIByL {
BarLCPass::BarLCPass() {
    frag = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/ssrt/"
        "ssrt_lc_debugger_frag.spv",
        {nullptr, RHI::ShaderStages::FRAGMENT});
    RDG::FullScreenPass::init(
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));

    uniform_buffer =
        GFX::GFXManager::get()->createStructuredUniformBuffer<Uniform>();
  }

auto BarLCPass::reflect() noexcept -> RDG::PassReflection {
    RDG::PassReflection reflector;

    reflector.addInput("DI")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(0, 1, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

    reflector.addInput("BaseColor")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(0, 1, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

    reflector.addInput("HiZ")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(0, RDG::MaxPossible, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

    reflector.addInput("NormalWS")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(0, 1, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

    reflector.addInput("ImportanceMIP")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(0, RDG::MaxPossible, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

    reflector.addInput("BoundingBoxMIP")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(0, RDG::MaxPossible, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

    reflector.addInput("BBNCPackMIP")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(0, RDG::MaxPossible, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

    reflector.addInput("NormalConeMIP")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(0, RDG::MaxPossible, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

    reflector.addOutput("Combined")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
        .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
        .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::ColorAttachment}
                     .setSubresource(0, 1, 0, 1)
                     .enableDepthWrite(false)
                     .setAttachmentLoc(0)
                     .setDepthCompareFn(RHI::CompareFunction::ALWAYS));

    return reflector;
}

auto BarLCPass::renderUI() noexcept -> void {
    {  // Select an item type
      const char* item_names[] = {"DI",
                                  "Specular",
                                  "Diffuse",
                                  "Debug Specular Ray",
                                  "Debug Occlusion Ray",
                                  "Show Normal Cone",
                                  "Show Tex Jacobian",
                                  "Visualize Importance"};
      int debug_mode = pConst.debug_mode;
      ImGui::Combo("Mode", &debug_mode, item_names, IM_ARRAYSIZE(item_names),
                   IM_ARRAYSIZE(item_names));
      pConst.debug_mode = uint32_t(debug_mode);
    }
    {  // Select an debug ray mode
      const char* item_names[] = {"HiZ", "DDA", "DDA_Con", "DDA_Tri0"};
      int debug_ray_mode = pConst.debug_ray_mode;
      ImGui::Combo("ScreenSpace Ray", &debug_ray_mode, item_names,
                   IM_ARRAYSIZE(item_names), IM_ARRAYSIZE(item_names));
      pConst.debug_ray_mode = uint32_t(debug_ray_mode);
    }
    if (pConst.debug_mode == 2 || pConst.debug_mode == 7) {
      int is_depth = pConst.is_depth;
      ImGui::DragInt("TexIS Depth", &is_depth, 1, 0, 11);
      pConst.is_depth = is_depth;
      {  // Select an item type
        const char* item_names[] = {"Luminance",
                                    "Luminance + d2 (Dachi)",
                                    "Luminance + G + d2 (Dachi)",
                                    "Luminance + NC",
                                    "Luminance + NC + d2 (Dachi)",
                                    "Luminance + NCP",
                                    "MIS Comp Heuristic"};
        int lightcut_mode = pConst.lightcut_mode;
        ImGui::Combo("Lightcut Mode", &lightcut_mode, item_names,
                     IM_ARRAYSIZE(item_names), IM_ARRAYSIZE(item_names));
        pConst.lightcut_mode = uint32_t(lightcut_mode);
      }
    }
    if (pConst.debug_mode >= 3 && pConst.debug_mode <= 4) {
      {
        float z_min = pConst.z_min;
        ImGui::DragFloat("Z Min", &z_min, 0.0001);
        pConst.z_min = z_min;
        float z_range = pConst.z_range;
        ImGui::DragFloat("Z Range", &z_range, 0.001);
        pConst.z_range = z_range;
      }
    }
    {
      int strategy = pConst.strategy;
      ImGui::DragInt("Strategy", &strategy, 1, 0, 4);
      pConst.strategy = strategy;
    }
    {
      int max_iteration = pConst.max_iteration;
      ImGui::DragInt("Max Iteration", &max_iteration, 1, 0, 10000);
      pConst.max_iteration = max_iteration;
    }
    {
      int mip_level = pConst.mip_level;
      ImGui::DragInt("MIP Level", &mip_level, 1, 0, pConst.hiz_mip_levels);
      pConst.mip_level = mip_level;
    }
    {
      int offset_steps = pConst.offset_steps;
      ImGui::DragInt("Offset cells", &offset_steps, 1, 0, 1000);
      pConst.offset_steps = offset_steps;
    }
    {
      float max_thickness = pConst.max_thickness;
      ImGui::DragFloat("Max Thickness", &max_thickness, 0.01);
      pConst.max_thickness = max_thickness;
    }
    {
      float z_clamper = pConst.z_clamper;
      ImGui::DragFloat("Z Clamper", &z_clamper, 0.01);
      pConst.z_clamper = z_clamper;
    }
    {
      float x = pConst.debugPos.x;
      float y = pConst.debugPos.y;
      float z = pConst.debugPos.z;
      float w = pConst.debugPos.w;
      ImGui::DragFloat("Debug x", &x, 1, 0, 1280 - 1);
      ImGui::DragFloat("Debug y", &y, 1, 0, 720 - 1);
      ImGui::DragFloat("Debug z", &z, 1, 0, 1280 - 1);
      ImGui::DragFloat("Debug w", &w, 1, 0, 720 - 1);
      pConst.debugPos.x = x;
      pConst.debugPos.y = y;
      pConst.debugPos.z = z;
      pConst.debugPos.w = w;
    }
}

auto BarLCPass::onInteraction(Platform::Input* input,
                              Editor::Widget::WidgetInfo* info) noexcept
    -> void {
    if (info->isFocused && info->isHovered) {
      if (pConst.debug_mode == 3 || pConst.debug_mode == 7) {
        // If left button is pressed
        if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
          pConst.debugPos.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
          pConst.debugPos.y = std::clamp(info->mousePos.y, 0.f, 719.f);
        }
      } else if (pConst.debug_mode == 4) {
        static bool firstClick = false;
        if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
          if (firstClick == false) {
            pConst.debugPos.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
            pConst.debugPos.y = std::clamp(info->mousePos.y, 0.f, 719.f);
            firstClick = true;
          } else {
            pConst.debugPos.z = std::clamp(info->mousePos.x, 0.f, 1279.f);
            pConst.debugPos.w = std::clamp(info->mousePos.y, 0.f, 719.f);
          }
        } else {
          if (firstClick) firstClick = false;
        }
      }
    }
}

auto BarLCPass::execute(RDG::RenderContext* context,
                        RDG::RenderData const& renderData) noexcept -> void {
    GFX::Texture* di = renderData.getTexture("DI");
    GFX::Texture* base_color = renderData.getTexture("BaseColor");
    GFX::Texture* hi_z = renderData.getTexture("HiZ");
    GFX::Texture* normalWS = renderData.getTexture("NormalWS");

    GFX::Texture* importance_mip = renderData.getTexture("ImportanceMIP");
    GFX::Texture* boundingbox_mip = renderData.getTexture("BoundingBoxMIP");
    GFX::Texture* bbncpack_mip = renderData.getTexture("BBNCPackMIP");
    GFX::Texture* normalcone_mip = renderData.getTexture("NormalConeMIP");

    GFX::Texture* out = renderData.getTexture("Combined");

    renderPassDescriptor = {
        {RHI::RenderPassColorAttachment{out->getRTV(0, 0, 1),
                                        nullptr,
                                        {0, 0, 0, 1},
                                        RHI::LoadOp::CLEAR,
                                        RHI::StoreOp::STORE}},
        RHI::RenderPassDepthStencilAttachment{},
    };

    if (hi_lumin_sampler == nullptr) {
      Core::GUID hil_sampler, hiz_sampler, basecolor;
      RHI::SamplerDescriptor hil_desc, hiz_desc, basecolor_desc;
      hil_desc.maxLod = importance_mip->texture->mipLevelCount();
      hil_desc.magFilter = RHI::FilterMode::NEAREST;
      hil_desc.minFilter = RHI::FilterMode::NEAREST;
      hil_desc.mipmapFilter = RHI::MipmapFilterMode::NEAREST;
      hiz_desc.maxLod = hi_z->texture->mipLevelCount();
      hiz_desc.magFilter = RHI::FilterMode::NEAREST;
      hiz_desc.minFilter = RHI::FilterMode::NEAREST;
      hiz_desc.mipmapFilter = RHI::MipmapFilterMode::NEAREST;
      basecolor_desc.magFilter = RHI::FilterMode::LINEAR;
      basecolor_desc.minFilter = RHI::FilterMode::LINEAR;
      hil_sampler =
          Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
      hiz_sampler =
          Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
      basecolor =
          Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
      GFX::GFXManager::get()->registerSamplerResource(hil_sampler, hil_desc);
      GFX::GFXManager::get()->registerSamplerResource(hiz_sampler, hiz_desc);
      GFX::GFXManager::get()->registerSamplerResource(basecolor,
                                                      basecolor_desc);
      hi_lumin_sampler =
          Core::ResourceManager::get()->getResource<GFX::Sampler>(hil_sampler);
      hi_z_sampler =
          Core::ResourceManager::get()->getResource<GFX::Sampler>(hiz_sampler);
      basecolor_sampler =
          Core::ResourceManager::get()->getResource<GFX::Sampler>(basecolor);
    }

    std::vector<RHI::BindGroupEntry>* set_0_entries =
        renderData.getBindGroupEntries("CommonScene");
    auto entry = (*set_0_entries)[0];
    entry.binding = 9;
    // getBindGroup(context, 0)->updateBinding({ entry });

    RHI::RenderPassEncoder* encoder = beginPass(context, out);

    RHI::Sampler* defaultSampler = GFX::GFXManager::get()->samplerTable.fetch(
        RHI::AddressMode::CLAMP_TO_EDGE, RHI::FilterMode::LINEAR,
        RHI::MipmapFilterMode::LINEAR);

    {
      std::vector<RHI::BindGroupEntry>* set_0_entries =
          renderData.getBindGroupEntries("CommonScene");
      pConst.view_size = Math::vec2(base_color->texture->width(),
                                    base_color->texture->height());
      pConst.hiz_mip_levels = hi_z->texture->mipLevelCount();
      pConst.sample_batch = renderData.getUInt("AccumIdx");

      SRenderer::CameraData* cd = reinterpret_cast<SRenderer::CameraData*>(
          renderData.getPtr("CameraData"));
      pConst.InvProjMat = Math::inverse((cd->projMat));
      pConst.ProjMat = cd->projMat;
      pConst.TransInvViewMat = Math::transpose(Math::inverse(cd->viewMat));
    }

    uniform_buffer.setStructure(pConst, context->flightIdx);

    getBindGroup(context, 0)
        ->updateBinding(std::vector<RHI::BindGroupEntry>{
            RHI::BindGroupEntry{
                0, RHI::BindingResource(base_color->getSRV(0, 1, 0, 1),
                                        basecolor_sampler->sampler.get())},
            RHI::BindGroupEntry{
                1, RHI::BindingResource(
                       hi_z->getSRV(0, hi_z->texture->mipLevelCount(), 0, 1),
                       hi_z_sampler->sampler.get())},
            RHI::BindGroupEntry{
                2, RHI::BindingResource(normalWS->getSRV(0, 1, 0, 1),
                                        defaultSampler)},
            RHI::BindGroupEntry{
                3, RHI::BindingResource(
                       importance_mip->getSRV(
                           0, importance_mip->texture->mipLevelCount(), 0, 1),
                       hi_lumin_sampler->sampler.get())},
            RHI::BindGroupEntry{
                4, RHI::BindingResource(
                       boundingbox_mip->getSRV(
                           0, importance_mip->texture->mipLevelCount(), 0, 1),
                       hi_lumin_sampler->sampler.get())},
            RHI::BindGroupEntry{
                5, RHI::BindingResource(
                       bbncpack_mip->getSRV(
                           0, importance_mip->texture->mipLevelCount(), 0, 1),
                       hi_lumin_sampler->sampler.get())},
            RHI::BindGroupEntry{
                6, RHI::BindingResource(
                       normalcone_mip->getSRV(
                           0, importance_mip->texture->mipLevelCount(), 0, 1),
                       hi_lumin_sampler->sampler.get())},
            RHI::BindGroupEntry{
                7, RHI::BindingResource(di->getSRV(0, 1, 0, 1),
                                        basecolor_sampler->sampler.get())},
            RHI::BindGroupEntry{
                8, RHI::BindingResource{uniform_buffer.getBufferBinding(
                       context->flightIdx)}},
            entry});

    // encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::FRAGMENT, 0,
    // sizeof(PushConstant));

    dispatchFullScreen(context);

    encoder->end();
}
}  // namespace SIByL