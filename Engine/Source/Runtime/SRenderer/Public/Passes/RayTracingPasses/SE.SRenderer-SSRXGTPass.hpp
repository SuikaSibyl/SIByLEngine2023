#include <SE.Editor.Core.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.RHI.hpp>
#include <SE.SRenderer.hpp>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <typeinfo>
#include "../../../../Application/Public/SE.Application.Config.h"

import SE.Platform.Window;

namespace SIByL::SRP {
SE_EXPORT struct SSRXGTPass : public RDG::RayTracingPass {
  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
    uint32_t primitive_num;
    uint32_t geo_setting;
  } pConst;

  Core::GUID rgen;
  Core::GUID rchit;
  Core::GUID rmiss;

  GFX::Buffer* vb;
  GFX::Buffer* ib;
  RHI::TLAS* tlas;

  SSRXGTPass() {
    rgen = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/ssrgt/"
        "ssrgt_rgen.spv",
        {nullptr, RHI::ShaderStages::RAYGEN});
    rchit = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/ssrgt/"
        "ssrgt_rchit.spv",
        {nullptr, RHI::ShaderStages::CLOSEST_HIT});
    rmiss = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/ssrgt/"
        "ssrgt_rmiss.spv",
        {nullptr, RHI::ShaderStages::MISS});

    GFX::SBTsDescriptor sbt = GFX::SBTsDescriptor{
        GFX::SBTsDescriptor::RayGenerationSBT{
            {{Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                rgen)}}},
        GFX::SBTsDescriptor::MissSBT{{
            {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                rmiss)},
        }},
        GFX::SBTsDescriptor::HitGroupSBT{{
            {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
                rchit)},
        }},
        GFX::SBTsDescriptor::CallableSBT{{}},
    };
    RayTracingPass::init(sbt, 3);

    uniform_buffer =
        GFX::GFXManager::get()->createStructuredUniformBuffer<Uniform>();
  }

  virtual auto reflect() noexcept -> RDG::PassReflection override {
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

    return reflector;
  }

  GFX::Sampler* hi_lumin_sampler = nullptr;
  GFX::Sampler* hi_z_sampler = nullptr;
  GFX::Sampler* basecolor_sampler = nullptr;

  struct alignas(64) Uniform {
    Math::vec2 view_size;
    int hiz_mip_levels;
    uint32_t max_iteration = 100;
    int strategy = 0;
    int sample_batch;
    uint32_t debug_ray_mode = 0;
    float max_thickness = 0.001;
    uint32_t debug_mode = 0;
    int32_t mip_level = 2;
    int32_t offset_steps = 2;
    float z_clamper = 1.0;

    Math::vec4 debugPos;
    float z_min = 0.00211;
    float z_range = 0.036;
    int is_depth = 11;
    int lightcut_mode = 0;

    Math::mat4 InvProjMat;
    Math::mat4 ProjMat;
    Math::mat4 TransInvViewMat;
  } pUniform;

  GFX::StructuredUniformBufferView<Uniform> uniform_buffer;

  virtual auto renderUI() noexcept -> void override {
    {  // Select an item type
      const char* item_names[] = {"DI",
                                  "Specular",
                                  "Diffuse",
                                  "Debug Specular Ray",
                                  "Debug Occlusion Ray",
                                  "Show Normal Cone",
                                  "Show Tex Jacobian",
                                  "Visualize Importance"};
      int debug_mode = pUniform.debug_mode;
      ImGui::Combo("Mode", &debug_mode, item_names, IM_ARRAYSIZE(item_names),
                   IM_ARRAYSIZE(item_names));
      pUniform.debug_mode = uint32_t(debug_mode);
    }
    {  // Select an debug ray mode
      const char* item_names[] = {"HiZ", "DDA"};
      int debug_ray_mode = pUniform.debug_ray_mode;
      ImGui::Combo("ScreenSpace Ray", &debug_ray_mode, item_names,
                   IM_ARRAYSIZE(item_names), IM_ARRAYSIZE(item_names));
      pUniform.debug_ray_mode = uint32_t(debug_ray_mode);
    }
    if (pUniform.debug_mode == 2 || pUniform.debug_mode == 7) {
      int is_depth = pUniform.is_depth;
      ImGui::DragInt("TexIS Depth", &is_depth, 1, 0, 11);
      pUniform.is_depth = is_depth;
      {  // Select an item type
        const char* item_names[] = {"Luminance",
                                    "Luminance + d2 (Dachi)",
                                    "Luminance + G + d2 (Dachi)",
                                    "Luminance + NC",
                                    "Luminance + NC + d2 (Dachi)",
                                    "Luminance + NCP",
                                    "MIS Comp Heuristic"};
        int lightcut_mode = pUniform.lightcut_mode;
        ImGui::Combo("Lightcut Mode", &lightcut_mode, item_names,
                     IM_ARRAYSIZE(item_names), IM_ARRAYSIZE(item_names));
        pUniform.lightcut_mode = uint32_t(lightcut_mode);
      }
    }
    if (pUniform.debug_mode >= 3 && pUniform.debug_mode <= 4) {
      {
        float z_min = pUniform.z_min;
        ImGui::DragFloat("Z Min", &z_min, 0.0001);
        pUniform.z_min = z_min;
        float z_range = pUniform.z_range;
        ImGui::DragFloat("Z Range", &z_range, 0.001);
        pUniform.z_range = z_range;
      }
    }
    {
      int strategy = pUniform.strategy;
      ImGui::DragInt("Strategy", &strategy, 1, 0, 4);
      pUniform.strategy = strategy;
    }
    {
      int max_iteration = pUniform.max_iteration;
      ImGui::DragInt("Max Iteration", &max_iteration, 1, 0, 1000);
      pUniform.max_iteration = max_iteration;
    }
    {
      int mip_level = pUniform.mip_level;
      ImGui::DragInt("MIP Level", &mip_level, 1, 0, pUniform.hiz_mip_levels);
      pUniform.mip_level = mip_level;
    }
    {
      int offset_steps = pUniform.offset_steps;
      ImGui::DragInt("Offset cells", &offset_steps, 1, 0, 1000);
      pUniform.offset_steps = offset_steps;
    }
    {
      float max_thickness = pUniform.max_thickness;
      ImGui::DragFloat("Max Thickness", &max_thickness, 0.01);
      pUniform.max_thickness = max_thickness;
    }
    {
      float z_clamper = pUniform.z_clamper;
      ImGui::DragFloat("Z Clamper", &z_clamper, 0.01);
      pUniform.z_clamper = z_clamper;
    }
    {
      float x = pUniform.debugPos.x;
      float y = pUniform.debugPos.y;
      float z = pUniform.debugPos.z;
      float w = pUniform.debugPos.w;
      ImGui::DragFloat("Debug x", &x, 1, 0, 1280 - 1);
      ImGui::DragFloat("Debug y", &y, 1, 0, 720 - 1);
      ImGui::DragFloat("Debug z", &z, 1, 0, 1280 - 1);
      ImGui::DragFloat("Debug w", &w, 1, 0, 720 - 1);
      pUniform.debugPos.x = x;
      pUniform.debugPos.y = y;
      pUniform.debugPos.z = z;
      pUniform.debugPos.w = w;
    }
  }

  virtual auto onInteraction(Platform::Input* input,
                             Editor::Widget::WidgetInfo* info) noexcept
      -> void override {
    if (info->isFocused && info->isHovered) {
      if (pUniform.debug_mode == 3 || pUniform.debug_mode == 7) {
        // If left button is pressed
        if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
          pUniform.debugPos.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
          pUniform.debugPos.y = std::clamp(info->mousePos.y, 0.f, 719.f);
        }
      } else if (pUniform.debug_mode == 4) {
        static bool firstClick = false;
        if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
          if (firstClick == false) {
            pUniform.debugPos.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
            pUniform.debugPos.y = std::clamp(info->mousePos.y, 0.f, 719.f);
            firstClick = true;
          } else {
            pUniform.debugPos.z = std::clamp(info->mousePos.x, 0.f, 1279.f);
            pUniform.debugPos.w = std::clamp(info->mousePos.y, 0.f, 719.f);
          }
        } else {
          if (firstClick) firstClick = false;
        }
      }
    }
  }

  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override {
    GFX::Texture* color = renderData.getTexture("Color");

    GFX::Texture* di = renderData.getTexture("DI");
    GFX::Texture* base_color = renderData.getTexture("BaseColor");
    GFX::Texture* hi_z = renderData.getTexture("HiZ");
    GFX::Texture* normalWS = renderData.getTexture("NormalWS");

    GFX::Texture* importance_mip = renderData.getTexture("ImportanceMIP");
    GFX::Texture* boundingbox_mip = renderData.getTexture("BoundingBoxMIP");
    GFX::Texture* bbncpack_mip = renderData.getTexture("BBNCPackMIP");
    GFX::Texture* normalcone_mip = renderData.getTexture("NormalConeMIP");

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
		hil_sampler = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
		hiz_sampler = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
		basecolor = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
		GFX::GFXManager::get()->registerSamplerResource(hil_sampler, hil_desc);
		GFX::GFXManager::get()->registerSamplerResource(hiz_sampler, hiz_desc);
		GFX::GFXManager::get()->registerSamplerResource(basecolor, basecolor_desc);
		hi_lumin_sampler = Core::ResourceManager::get()->getResource<GFX::Sampler>(hil_sampler);
		hi_z_sampler = Core::ResourceManager::get()->getResource<GFX::Sampler>(hiz_sampler);
		basecolor_sampler = Core::ResourceManager::get()->getResource<GFX::Sampler>(basecolor);
	}

    RHI::Sampler* sampler = GFX::GFXManager::get()->samplerTable.fetch(
        RHI::AddressMode::CLAMP_TO_EDGE, RHI::FilterMode::LINEAR,
        RHI::MipmapFilterMode::LINEAR);

	{
		std::vector<RHI::BindGroupEntry>* set_0_entries = renderData.getBindGroupEntries("CommonScene");
		pUniform.view_size = Math::vec2(base_color->texture->width(), base_color->texture->height());
		pUniform.hiz_mip_levels = hi_z->texture->mipLevelCount();
		pUniform.sample_batch = renderData.getUInt("AccumIdx");

		SRenderer::CameraData* cd = reinterpret_cast<SRenderer::CameraData*>(renderData.getPtr("CameraData"));
		pUniform.InvProjMat = Math::inverse((cd->projMat));
		pUniform.ProjMat = cd->projMat;
		pUniform.TransInvViewMat = Math::transpose(Math::inverse(cd->viewMat));
	}

	uniform_buffer.setStructure(pUniform, context->flightIdx);

	getBindGroup(context, 1)->updateBinding(std::vector<RHI::BindGroupEntry>{
		RHI::BindGroupEntry{ 0,RHI::BindingResource(base_color->getSRV(0,1,0,1), basecolor_sampler->sampler.get()) },
		RHI::BindGroupEntry{ 1,RHI::BindingResource(hi_z->getSRV(0,hi_z->texture->mipLevelCount(),0,1), hi_z_sampler->sampler.get()) },
		RHI::BindGroupEntry{ 2,RHI::BindingResource(normalWS->getSRV(0,1,0,1), sampler) },
		RHI::BindGroupEntry{ 3,RHI::BindingResource(importance_mip ->getSRV(0,importance_mip->texture->mipLevelCount(),0,1), hi_lumin_sampler->sampler.get()) },
		RHI::BindGroupEntry{ 4,RHI::BindingResource(boundingbox_mip->getSRV(0,importance_mip->texture->mipLevelCount(),0,1), hi_lumin_sampler->sampler.get()) },
		RHI::BindGroupEntry{ 5,RHI::BindingResource(bbncpack_mip ->getSRV(0,importance_mip->texture->mipLevelCount(),0,1), hi_lumin_sampler->sampler.get()) },
		RHI::BindGroupEntry{ 6,RHI::BindingResource(normalcone_mip->getSRV(0,importance_mip->texture->mipLevelCount(),0,1), hi_lumin_sampler->sampler.get()) },
		RHI::BindGroupEntry{ 7,RHI::BindingResource(di->getSRV(0,1,0,1), basecolor_sampler->sampler.get()) },
		RHI::BindGroupEntry{ 8,RHI::BindingResource{uniform_buffer.getBufferBinding(context->flightIdx)} },
	});

    std::vector<RHI::BindGroupEntry>* set_0_entries =
        renderData.getBindGroupEntries("CommonScene");
    getBindGroup(context, 0)->updateBinding({
      (*set_0_entries)[0],
      RHI::BindGroupEntry{1, RHI::BufferBinding{vb->buffer.get(), 0,
                                              vb->buffer->size()}},
      RHI::BindGroupEntry{2, RHI::BufferBinding{ib->buffer.get(), 0,
                                              ib->buffer->size()}},
      RHI::BindGroupEntry{3, tlas},
      RHI::BindGroupEntry{4, RHI::BindingResource{color->getUAV(0, 0, 1)}}});

    RHI::RayTracingPassEncoder* encoder = beginPass(context);

    uint32_t batchIdx = renderData.getUInt("AccumIdx");

    pConst.width = color->texture->width();
    pConst.height = color->texture->height();
    pConst.sample_batch = batchIdx;

    encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                           sizeof(PushConstant));
    encoder->traceRays(color->texture->width(), color->texture->height(), 1);

    encoder->end();
  }
};
}  // namespace SIByL::SRP