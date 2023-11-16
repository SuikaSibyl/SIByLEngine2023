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
SE_EXPORT struct MIPMinPoolingSubPass : public RDG::FullScreenPass {
  MIPMinPoolingSubPass(size_t mip_offset) : mipOffset(mip_offset) {
    frag = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/"
        "min_mip_float32_frag.spv",
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

    int32_t src_width = target->getRTV(mipOffset, 0, 1)->getWidth();
    int32_t src_height = target->getRTV(mipOffset, 0, 1)->getHeight();

    struct PushConstant {
      int32_t src_width;
      int32_t src_height;
      int32_t dst_width;
      int32_t dst_height;
    } ps = {src_width, src_height, std::max(src_width >> 1, 1),
            std::max(src_height >> 1, 1)};

    RHI::RenderPassEncoder* encoder =
        beginPass(context, ps.dst_width, ps.dst_height);
    encoder->pushConstants(&ps, (uint32_t)RHI::ShaderStages::FRAGMENT, 0,
                           sizeof(PushConstant));

    dispatchFullScreen(context);

    encoder->end();
  }

  size_t mipOffset;
  Core::GUID frag;
};

struct MIPMinInputDummy : public RDG::DummyPass {
  MIPMinInputDummy() { RDG::Pass::init(); }

  virtual auto reflect() noexcept -> RDG::PassReflection {
    RDG::PassReflection reflector;

    reflector.addInputOutput("F32").isTexture();

    reflector.addOutput("F32MIP")
        .isTexture()
        .withSizeRelative("F32")
        .withLevels(RDG::MaxPossible)
        .withFormat(RHI::TextureFormat::R32_FLOAT);

    return reflector;
  }
};

SE_EXPORT struct MIPMinPoolingPass : public RDG::Subgraph {
  MIPMinPoolingPass(size_t width, size_t height)
      : width(width), height(height) {
    mipCount = std::log2(std::max(width, height)) + 1;
  }

  virtual auto alias() noexcept -> RDG::AliasDict override {
    RDG::AliasDict dict;
    dict.addAlias("Input", CONCAT("Input Dummy"), "F32");
    dict.addAlias("Output",
                  CONCAT("MinPool Pass " + std::to_string(mipCount - 2)),
                  "R32");
    // dict.addAlias("Input", CONCAT("Input Dummy"), "F32");
    // dict.addAlias("Output", CONCAT("Blit Pass"), "Traget");
    return dict;
  }

  virtual auto onRegister(RDG::Graph* graph) noexcept -> void override {
    graph->addPass(std::make_unique<MIPMinInputDummy>(), CONCAT("Input Dummy"));
    graph->addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0}),
                   CONCAT("Blit Pass"));
    graph->addPass(std::make_unique<MIPMinPoolingSubPass>(0),
                   CONCAT("MinPool Pass " + std::to_string(0)));

    graph->addEdge(CONCAT("Input Dummy"), "F32", CONCAT("Blit Pass"), "Source");
    graph->addEdge(CONCAT("Input Dummy"), "F32MIP", CONCAT("Blit Pass"),
                   "Target");
    graph->addEdge(CONCAT("Blit Pass"), "Target", CONCAT("MinPool Pass 0"),
                   "R32");

    for (size_t i = 1; i < mipCount - 1; ++i) {
      graph->addPass(std::make_unique<MIPMinPoolingSubPass>(i),
                     CONCAT("MinPool Pass " + std::to_string(i)));
      graph->addEdge(CONCAT("MinPool Pass " + std::to_string(i - 1)), "R32",
                     CONCAT("MinPool Pass " + std::to_string(i)), "R32");
    }
  }

  size_t mipCount;
  size_t width, height;
};
}  // namespace SIByL