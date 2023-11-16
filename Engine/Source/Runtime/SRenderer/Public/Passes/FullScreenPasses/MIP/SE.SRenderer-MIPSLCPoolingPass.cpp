#include "SE.SRenderer-MIPSLCPoolingPass.hpp"
#include "../SE.SRenderer-Blit.hpp"
#include "../SE.SRenderer-ClearPass.hpp"

namespace SIByL {
MIPSLCSubPass::MIPSLCSubPass(size_t mip_offset) : mipOffset(mip_offset) {
  frag = GFX::GFXManager::get()->registerShaderModuleResource(
      "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/rsm/"
      "rsm_slc_mipgen_frag.spv",
      {nullptr, RHI::ShaderStages::FRAGMENT});
  RDG::FullScreenPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto MIPSLCSubPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInputOutput("PixImportance")
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
              .addStage(
                  (uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT));

  reflector.addInputOutput("NormalCone")
      .isTexture()
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
              .setAttachmentLoc(1)
              .addStage(
                  (uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT));

  reflector.addInputOutput("AABBXY")
      .isTexture()
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
              .setAttachmentLoc(2)
              .addStage(
                  (uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT));

  reflector.addInputOutput("AABBZ")
      .isTexture()
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
              .setAttachmentLoc(3)
              .addStage(
                  (uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT));

  return reflector;
}

auto MIPSLCSubPass::execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* importance_mip = renderData.getTexture("PixImportance");
  GFX::Texture* normalcone = renderData.getTexture("NormalCone");
  GFX::Texture* aabbxy = renderData.getTexture("AABBXY");
  GFX::Texture* aabbz = renderData.getTexture("AABBZ");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{importance_mip->getRTV(mipOffset + 1, 0, 1),
           nullptr, {0, 0, 0, 1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE},
       RHI::RenderPassColorAttachment{normalcone->getRTV(mipOffset + 1, 0, 1),
           nullptr, {0, 0, 0, 1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE},
       RHI::RenderPassColorAttachment{aabbxy->getRTV(mipOffset + 1, 0, 1),
           nullptr, {0, 0, 0, 1},  RHI::LoadOp::CLEAR, RHI::StoreOp::STORE},
       RHI::RenderPassColorAttachment{aabbz->getRTV(mipOffset + 1, 0, 1),
           nullptr, {0, 0, 0, 1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{},
  };

  auto* defaul_sampler = GFX::GFXManager::get()->samplerTable.fetch(
      RHI::AddressMode::CLAMP_TO_EDGE, RHI::FilterMode::LINEAR,
      RHI::MipmapFilterMode::LINEAR);
  getBindGroup(context, 0)
      ->updateBinding(std::vector<RHI::BindGroupEntry>{
          {0, RHI::BindingResource(importance_mip->getSRV(mipOffset, 1, 0, 1), defaul_sampler)},
          {1, RHI::BindingResource(normalcone->getSRV(mipOffset, 1, 0, 1), defaul_sampler)},
          {2, RHI::BindingResource(aabbxy->getSRV(mipOffset, 1, 0, 1), defaul_sampler)},
          {3, RHI::BindingResource(aabbz->getSRV(mipOffset, 1, 0, 1), defaul_sampler)},
      });

  int32_t src_width = importance_mip->getRTV(mipOffset, 0, 1)->getWidth();
  int32_t src_height = importance_mip->getRTV(mipOffset, 0, 1)->getHeight();

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

MIPSLCPass::MIPSLCPass(size_t width, size_t height)
    : width(width), height(height) {
  mipCount = std::log2(std::max(width, height)) + 1;
}

auto MIPSLCPass::alias() noexcept -> RDG::AliasDict {
  RDG::AliasDict dict;
  dict.addAlias("PixImportance", CONCAT("SLCPool Pass 0"), "PixImportance");
  dict.addAlias("NormalCone", CONCAT("SLCPool Pass 0"), "NormalCone");
  dict.addAlias("AABBXY", CONCAT("SLCPool Pass 0"), "AABBXY");
  dict.addAlias("AABBZ", CONCAT("SLCPool Pass 0"), "AABBZ");

  dict.addAlias("PixImportanceOut",
                CONCAT("SLCPool Pass " + std::to_string(mipCount - 2)),
                "PixImportance");
  dict.addAlias("NormalConeOut",
                CONCAT("SLCPool Pass " + std::to_string(mipCount - 2)),
                "NormalCone");
  dict.addAlias("AABBXYOut",
                CONCAT("SLCPool Pass " + std::to_string(mipCount - 2)),
                "AABBXY");
  dict.addAlias("AABBZOut",
                CONCAT("SLCPool Pass " + std::to_string(mipCount - 2)),
                "AABBZ");
  return dict;
}

auto MIPSLCPass::onRegister(RDG::Graph* graph) noexcept -> void {

  graph->addPass(std::make_unique<MIPSLCSubPass>(0),
                 CONCAT("SLCPool Pass 0"));

  for (size_t i = 1; i < mipCount - 1; ++i) {
    graph->addPass(std::make_unique<MIPSLCSubPass>(i),
                   CONCAT("SLCPool Pass " + std::to_string(i)));

    graph->addEdge(CONCAT("SLCPool Pass " + std::to_string(i - 1)),
                   "PixImportance", CONCAT("SLCPool Pass " + std::to_string(i)),
                   "PixImportance");
    graph->addEdge(
        CONCAT("SLCPool Pass " + std::to_string(i - 1)), "NormalCone",
        CONCAT("SLCPool Pass " + std::to_string(i)), "NormalCone");
    graph->addEdge(CONCAT("SLCPool Pass " + std::to_string(i - 1)), "AABBXY",
                   CONCAT("SLCPool Pass " + std::to_string(i)), "AABBXY");
    graph->addEdge(CONCAT("SLCPool Pass " + std::to_string(i - 1)), "AABBZ",
                   CONCAT("SLCPool Pass " + std::to_string(i)), "AABBZ");
  }
}

MIPTiledVisInputPass::MIPTiledVisInputPass(uint32_t tile_size,
                                      uint32_t tile_buffer_size,
                                 uint32_t width, uint32_t height)
    : tile_size(tile_size),
      tile_buffer_size(tile_buffer_size),
      width(width),
      height(height) {
  RDG::Pass::init();
}

auto MIPTiledVisInputPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  uint32_t tile_x = (width + tile_size - 1) / tile_size;
  uint32_t tile_y = (height + tile_size - 1) / tile_size;
  uint32_t buffer_x = tile_x * tile_size;
  uint32_t buffer_y = tile_y * tile_size;
  uint32_t miplevel = std::log2(tile_size) + 1;

  reflector.addOutput("TiledVisibility")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT |
                  (uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
      .withSize(Math::ivec2(width, height))
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withLevels(miplevel);

  reflector.addOutput("InputVisibility")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT |
                  (uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withSize(Math::ivec2(width, height));

  reflector.addOutput("ImportanceSplatting")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT |
                  (uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
      .withFormat(RHI::TextureFormat::R32_FLOAT)
      .withSize(Math::ivec2(512, 512));

  return reflector;
}

MIPTiledVisSubPass::MIPTiledVisSubPass(size_t mip_offset)
    : mipOffset(mip_offset) {
  frag = GFX::GFXManager::get()->registerShaderModuleResource(
      "../Engine/Binaries/Runtime/spirv/SRenderer/raytracer/rsm/"
      "tv_mipgen_frag.spv",
      {nullptr, RHI::ShaderStages::FRAGMENT});
  RDG::FullScreenPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto MIPTiledVisSubPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInputOutput("TiledVisibility")
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
              .addStage(
                  (uint32_t)RHI::PipelineStages::COLOR_ATTACHMENT_OUTPUT_BIT));

  return reflector;
}

auto MIPTiledVisSubPass::execute(RDG::RenderContext* context,
                            RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* visibility_buffer = renderData.getTexture("TiledVisibility");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{
           visibility_buffer->getRTV(mipOffset + 1, 0, 1),
           nullptr,
           {0, 0, 0, 1},
           RHI::LoadOp::CLEAR,
           RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{},
  };

  auto* defaul_sampler = GFX::GFXManager::get()->samplerTable.fetch(
      RHI::AddressMode::CLAMP_TO_EDGE, RHI::FilterMode::LINEAR,
      RHI::MipmapFilterMode::LINEAR);
  getBindGroup(context, 0)
      ->updateBinding(std::vector<RHI::BindGroupEntry>{
          {0, RHI::BindingResource(visibility_buffer->getSRV(mipOffset, 1, 0, 1),
                                   defaul_sampler)},
      });

  int32_t src_width = visibility_buffer->getRTV(mipOffset, 0, 1)->getWidth();
  int32_t src_height = visibility_buffer->getRTV(mipOffset, 0, 1)->getHeight();

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

MIPTiledVisPass::MIPTiledVisPass(uint32_t tile_size,
                                   uint32_t tile_buffer_size, uint32_t width,
                                   uint32_t height)
    : width(width),
      height(height),
      tile_size(tile_size),
      tile_buffer_size(tile_buffer_size) {
  uint32_t tile_x = (width + tile_size - 1) / tile_size;
  uint32_t tile_y = (height + tile_size - 1) / tile_size;
  uint32_t buffer_x = tile_x * tile_size;
  uint32_t buffer_y = tile_y * tile_size;
  mipCount = std::log2(tile_size) + 1;
}

auto MIPTiledVisPass::alias() noexcept -> RDG::AliasDict {
  RDG::AliasDict dict;
  dict.addAlias("Input", CONCAT("Clear Visibility"), "Target");
  dict.addAlias("Output",
                CONCAT("TVMIP Pass " + std::to_string(mipCount - 2)),
                "TiledVisibility");
  return dict;
}

auto MIPTiledVisPass::onRegister(RDG::Graph* graph) noexcept -> void {
  graph->addPass(std::make_unique<MIPTiledVisInputPass>(
                     tile_size, tile_buffer_size, width, height),
                 CONCAT("TiledVis Dummy"));
  graph->addPass(std::make_unique<BlitPass>(BlitPass::Descriptor{0, 0, 0, 0}), 
                 CONCAT("Blit Visibility"));
  graph->addPass(std::make_unique<MIPTiledVisSubPass>(0), CONCAT("TVMIP Pass 0"));
  //ClearPassR32f
  graph->addEdge(CONCAT("TiledVis Dummy"), "InputVisibility",
                 CONCAT("Blit Visibility"), "Source");
  graph->addEdge(CONCAT("TiledVis Dummy"), "TiledVisibility",
                 CONCAT("Blit Visibility"), "Target");
  graph->addEdge(CONCAT("Blit Visibility"), "Target",
                 CONCAT("TVMIP Pass 0"), "TiledVisibility");

  graph->addPass(std::make_unique<ClearPassR32f>(), CONCAT("Clear Visibility"));
  graph->addEdge(CONCAT("Blit Visibility"), "Source",
                 CONCAT("Clear Visibility"), "Target");

  for (size_t i = 1; i < mipCount - 1; ++i) {
    graph->addPass(std::make_unique<MIPTiledVisSubPass>(i),
                   CONCAT("TVMIP Pass " + std::to_string(i)));

    graph->addEdge(CONCAT("TVMIP Pass " + std::to_string(i - 1)),
                   "TiledVisibility", CONCAT("TVMIP Pass " + std::to_string(i)),
                   "TiledVisibility");
  }
}
}