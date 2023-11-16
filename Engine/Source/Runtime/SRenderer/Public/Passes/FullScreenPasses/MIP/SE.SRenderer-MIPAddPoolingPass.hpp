#pragma once
#include <array>
#include <compare>
#include <filesystem>
#include <functional>
#include <memory>
#include <typeinfo>
#include <vector>
#include "../../../../../Application/Public/SE.Application.Config.h"
#include "../SE.SRenderer-Blit.hpp"

namespace SIByL {
SE_EXPORT struct MIPAddPoolingSubPass : public RDG::FullScreenPass {
  MIPAddPoolingSubPass(size_t mip_offset) : mipOffset(mip_offset) {
    frag = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/"
        "sumup_mip_float32_frag.spv",
        {nullptr, RHI::ShaderStages::FRAGMENT});
    RDG::FullScreenPass::init(
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
  }

  virtual auto reflect() noexcept -> RDG::PassReflection {
    RDG::PassReflection reflector;

    reflector.addInputOutput("R32")
        .isTexture()
        .withFormat(RHI::TextureFormat::R32_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT |
                    (uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(mipOffset, mipOffset + 1, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT))
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::ColorAttachment}
                .setSubresource(mipOffset + 1, mipOffset + 2, 0, 1)
                .setAttachmentLoc(0)
                .addStage((
                    uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT));

    return reflector;
  }

  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void {
    GFX::Texture* target = renderData.getTexture("R32");

    renderPassDescriptor = {
        {RHI::RenderPassColorAttachment{target->getRTV(mipOffset + 1, 0, 1),
                                        nullptr,
                                        {0, 0, 0, 1},
                                        RHI::LoadOp::CLEAR,
                                        RHI::StoreOp::STORE}},
        RHI::RenderPassDepthStencilAttachment{},
    };

    RHI::Sampler* sampler = GFX::GFXManager::get()->samplerTable.fetch(
        RHI::AddressMode::CLAMP_TO_EDGE, RHI::FilterMode::NEAREST,
        RHI::MipmapFilterMode::NEAREST);

    getBindGroup(context, 0)
        ->updateBinding(std::vector<RHI::BindGroupEntry>{
            {0, RHI::BindingResource(
                    std::vector<RHI::TextureView*>{
                        target->getSRV(mipOffset, 1, 0, 1)},
                                  sampler)}});

    size_t src_size = target->getRTV(mipOffset, 0, 1)->getWidth();

    RHI::RenderPassEncoder* encoder =
        beginPass(context, src_size / 2, src_size / 2);
    encoder->pushConstants(&src_size, (uint32_t)RHI::ShaderStages::FRAGMENT, 0,
                           sizeof(uint32_t));

    dispatchFullScreen(context);

    encoder->end();
  }

  size_t mipOffset;
  Core::GUID frag;
};

SE_EXPORT struct MIPAddPoolingPass : public RDG::Subgraph {
  MIPAddPoolingPass(size_t textureSize) : textureSize(textureSize) {
    mipCount = std::log2(textureSize) + 1;
  }

  virtual auto alias() noexcept -> RDG::AliasDict override {
    RDG::AliasDict dict;
    dict.addAlias("Input", CONCAT("Pre-Z Pass 0"), "R32");
    dict.addAlias("Output",
                  CONCAT("Pre-Z Pass " + std::to_string(mipCount - 2)), "R32");
    return dict;
  }

  virtual auto onRegister(RDG::Graph* graph) noexcept -> void override {
    for (size_t i = 0; i < mipCount - 1; ++i) {
      graph->addPass(std::make_unique<MIPAddPoolingSubPass>(i),
                     CONCAT("Pre-Z Pass " + std::to_string(i)));
      if (i != 0)
        graph->addEdge(CONCAT("Pre-Z Pass " + std::to_string(i - 1)), "R32",
                       CONCAT("Pre-Z Pass " + std::to_string(i)), "R32");
    }
  }

  size_t mipCount;
  size_t textureSize;
};
}  // namespace SIByL