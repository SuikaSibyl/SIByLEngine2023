#pragma once

#include <SE.Editor.Core.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.RHI.hpp>
#include <SE.SRenderer.hpp>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <typeinfo>
#include <bitset>
#include <random>

#include "../../../../Application/Public/SE.Application.Config.h"

namespace SIByL::SRP {
SE_EXPORT struct RSMShareInfo {
  Math::mat4 inv_view;
  Math::mat4 inv_proj;
  Math::mat4 proj_view;
  Math::vec3 direction;
  float area;
};

SE_EXPORT struct DirectRSMPass : public RDG::RayTracingPass {

  struct PushConstant {
    Math::mat4 inv_view;
    Math::mat4 inv_proj;
    Math::vec3 direction;
    uint32_t useWeight;
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
  };

  float scaling_x = 1.f;
  float scaling_y = 1.f;
  bool use_weight = false;

  uint32_t width, height;
  Core::GUID rsm_rgen;
  RSMShareInfo* info = nullptr;

  DirectRSMPass(uint32_t width, uint32_t height, RSMShareInfo* info);

  virtual auto reflect() noexcept -> RDG::PassReflection override;

  virtual auto renderUI() noexcept -> void override;

  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override;
};

SE_EXPORT struct RSMGIPass : public RDG::RayTracingPass {
  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
    uint8_t mode = 1;
    uint8_t samplingStrategy = 0;
    uint8_t miplayer = 9;
    uint8_t pxStrategy = 0;
    uint32_t debug_info_x;
    uint32_t debug_info_y;
    float imp_vis_scalar = 1.f;
    uint32_t slc_use_normalcone : 1 = 0;
    uint32_t slc_use_visibility : 1 = 0;
    uint32_t slc_use_importance : 1 = 0;
    uint32_t slc_mode_padding   : 5 = 0;
    uint32_t visualize_ideal    : 1 = 0;
    uint32_t visualize_mode     : 7 = 0;
    uint32_t slc_mode           : 16 = 0;
  } pConst;

  RSMShareInfo* info;
  struct alignas(64) Uniform {
    Math::mat4 inv_view;
    Math::mat4 inv_proj;
    Math::mat4 proj_view;
    Math::vec3 direction;
    float area;
  };

  GFX::StructuredUniformBufferView<Uniform> uniform_buffer;

  RSMGIPass(RSMShareInfo* info) : info(info) {
    udpt_rgen = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/rsm/"
        "rsm_gi_rgen.spv",
        {nullptr, RHI::ShaderStages::RAYGEN});

    GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
    sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
        {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(
            udpt_rgen)}};

    uniform_buffer =
        GFX::GFXManager::get()->createStructuredUniformBuffer<Uniform>();

    RayTracingPass::init(sbt, 3);
  }

  auto renderUI() noexcept -> void {
    {  // Select an item type
      const char* item_names[] = {
          "Combined",
          "DI",
          "GI",
      };
      int debug_mode = pConst.mode;
      ImGui::Combo("Vis Mode", &debug_mode, item_names, IM_ARRAYSIZE(item_names),
                   IM_ARRAYSIZE(item_names));
      pConst.mode = uint8_t(debug_mode);
    }
    {  // Select an item type
      const char* item_names[] = {
          "Cos Hemisphere",
          "Pixel Selection",
          "Pixel Selection RIS",
      };
      int sample_mode = pConst.samplingStrategy;
      ImGui::Combo("Sample Mode", &sample_mode, item_names,
                   IM_ARRAYSIZE(item_names),
                   IM_ARRAYSIZE(item_names));
      pConst.samplingStrategy = uint8_t(sample_mode);
    }
    {  // Select an item type
      const char* item_names[] = {
          "Uniform",
          "Irradiance",
          "SLC",
      };
      int ps_mode = pConst.pxStrategy;
      ImGui::Combo("Pixel Sample Mode", &ps_mode, item_names,
                   IM_ARRAYSIZE(item_names), IM_ARRAYSIZE(item_names));
      pConst.pxStrategy = uint8_t(ps_mode);
    }
    {
      int miplayer = pConst.miplayer;
      ImGui::DragInt("TexIS Depth", &miplayer, 1, 0, 9);
      pConst.miplayer = miplayer;
    }
    {
      float imp_vis_scalar = pConst.imp_vis_scalar;
      ImGui::DragFloat("Vis Scalar", &imp_vis_scalar, 0.01);
      pConst.imp_vis_scalar = imp_vis_scalar;
    }
    {
      bool UseNormalCone = pConst.slc_use_normalcone;
      ImGui::Checkbox("Use Normal Cone", &UseNormalCone);
      pConst.slc_use_normalcone = UseNormalCone;
      bool UseVisibility = pConst.slc_use_visibility;
      ImGui::Checkbox("Use Visibility", &UseVisibility);
      pConst.slc_use_visibility = UseVisibility;
      bool UseImportance = pConst.slc_use_importance;
      ImGui::Checkbox("Use Importance", &UseImportance);
      pConst.slc_use_importance = UseImportance;
    }
    {
      bool visualize_ideal = pConst.visualize_ideal;
      ImGui::Checkbox("Visualize Ideal", &visualize_ideal);
      pConst.visualize_ideal = visualize_ideal;
    }
  }

  auto onInteraction(Platform::Input* input,
                                Editor::Widget::WidgetInfo* info) noexcept
      -> void {
    if (info->isFocused && info->isHovered) {
      // If left button is pressed
      if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
        pConst.debug_info_x = std::clamp(info->mousePos.x, 0.f, 1279.f);
        pConst.debug_info_y = std::clamp(info->mousePos.y, 0.f, 719.f);
      }
    }
  }

  virtual auto reflect() noexcept -> RDG::PassReflection override {
    RDG::PassReflection reflector;

    reflector.addInput("PixImportance")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(0, RDG::MaxPossible, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

    reflector.addInput("NormalCone")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(0, RDG::MaxPossible, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

    reflector.addInput("AABBXY")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(0, RDG::MaxPossible, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

    reflector.addInput("AABBZ")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(0, RDG::MaxPossible, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

    reflector.addInput("TVIn")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(0, 5, 0, 1)
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

    reflector.addInputOutput("TVOut")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::StorageBinding}
                .setSubresource(0, 1, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

    reflector.addOutput("DebugImportance")
        .isTexture()
        .withSize(Math::ivec3(512, 512, 1))
        .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::StorageBinding}
                .addStage(
                    (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

    reflector.addInput("ImportanceSplatting")
        .isTexture()
        .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::StorageBinding}
                .addStage(
                    (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

    return reflector;
  }

  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept
      -> void override {
    GFX::Texture* importance = renderData.getTexture("PixImportance");
    GFX::Texture* normalcone = renderData.getTexture("NormalCone");
    GFX::Texture* aabbxy = renderData.getTexture("AABBXY");
    GFX::Texture* aabbz = renderData.getTexture("AABBZ");
    GFX::Texture* color = renderData.getTexture("Color");
    GFX::Texture* tvin = renderData.getTexture("TVIn");
    GFX::Texture* tvout = renderData.getTexture("TVOut");
    GFX::Texture* debugimp = renderData.getTexture("DebugImportance");
    GFX::Texture* impsplat = renderData.getTexture("ImportanceSplatting");

    Uniform uniform{
      info->inv_view,
      info->inv_proj,
      info->proj_view,
      info->direction,
      info->area
    };
    uniform_buffer.setStructure(uniform, context->flightIdx);

    auto* defaul_sampler = GFX::GFXManager::get()->samplerTable.fetch(
        RHI::AddressMode::CLAMP_TO_EDGE, RHI::FilterMode::LINEAR,
        RHI::MipmapFilterMode::LINEAR);

    std::vector<RHI::BindGroupEntry>* set_0_entries =
        renderData.getBindGroupEntries("CommonScene");
    getBindGroup(context, 0)->updateBinding(*set_0_entries);
    std::vector<RHI::BindGroupEntry>* set_1_entries =
        renderData.getBindGroupEntries("CommonRT");
    getBindGroup(context, 1)->updateBinding(*set_1_entries);
    getBindGroup(context, 2)->updateBinding({
      RHI::BindGroupEntry{0, RHI::BindingResource(
        importance->getSRV(0, importance->texture->mipLevelCount(), 0, 1),
        defaul_sampler)},
      RHI::BindGroupEntry{1, RHI::BindingResource(
        normalcone->getSRV(0, normalcone->texture->mipLevelCount(), 0, 1),
        defaul_sampler)},
      RHI::BindGroupEntry{2, RHI::BindingResource(
        aabbxy->getSRV(0, aabbxy->texture->mipLevelCount(), 0, 1),
        defaul_sampler)},
      RHI::BindGroupEntry{3, RHI::BindingResource(
        aabbz->getSRV(0, aabbz->texture->mipLevelCount(), 0, 1),
        defaul_sampler)},
      RHI::BindGroupEntry{4, RHI::BindingResource(
        tvin->getSRV(0, tvin->texture->mipLevelCount(), 0, 1),
        defaul_sampler)},
      RHI::BindGroupEntry{5, RHI::BindingResource{color->getUAV(0, 0, 1)}},
      RHI::BindGroupEntry{6, RHI::BindingResource{debugimp->getUAV(0, 0, 1)}},
      RHI::BindGroupEntry{7, RHI::BindingResource{tvout->getUAV(0, 0, 1)}},
      RHI::BindGroupEntry{8, RHI::BindingResource{uniform_buffer.getBufferBinding(context->flightIdx)}},
      RHI::BindGroupEntry{9, RHI::BindingResource{impsplat->getUAV(0, 0, 1)}},
    });

    RHI::RayTracingPassEncoder* encoder = beginPass(context);

    uint32_t batchIdx = renderData.getUInt("AccumIdx");

    static std::random_device rd;   // a seed source for the random number engine
    static std::mt19937 gen(rd());  // mersenne_twister_engine seeded with rd()
    static std::uniform_int_distribution<uint32_t> distrib(
        0, std::numeric_limits<uint32_t>::max());

    pConst.width = color->texture->width();
    pConst.height = color->texture->height();
    pConst.sample_batch = distrib(gen);

    encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                           sizeof(PushConstant));
    encoder->traceRays(color->texture->width(), color->texture->height(), 1);

    encoder->end();
  }

  Core::GUID udpt_rgen;
};
}  // namespace SIByL::SRP