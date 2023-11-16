#pragma once
#include "SE.SRenderer-Blit.hpp"
#include <array>
#include <compare>
#include <filesystem>
#include <functional>
#include <memory>
#include <typeinfo>
#include "../../../../Application/Public/SE.Application.Config.h"

namespace SIByL {
std::string to_string(BlitPass::SourceType type) {
  switch (type) {
    case SIByL::BlitPass::SourceType::UINT:
      return "uint";
    case SIByL::BlitPass::SourceType::UINT2:
      return "uint2";
    case SIByL::BlitPass::SourceType::UINT3:
      return "uint3";
    case SIByL::BlitPass::SourceType::UINT4:
      return "uint4";
    case SIByL::BlitPass::SourceType::FLOAT:
      return "float";
    case SIByL::BlitPass::SourceType::FLOAT2:
      return "float2";
    case SIByL::BlitPass::SourceType::FLOAT3:
      return "float3";
    case SIByL::BlitPass::SourceType::FLOAT4:
      return "float4";
    default:
      return "undefined";
  }
}

BlitPass::BlitPass(Descriptor const& desc) : desc(desc) {
  std::string type_string = to_string(desc.sourceType);
  auto [frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/fullscreen/"
      "blit.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      },
      {std::make_pair("RESOURCE_TYPE", type_string.c_str())});
  RDG::FullScreenPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto BlitPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInput("Source")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::TextureBinding}
              .setSubresource(desc.src_mip, desc.src_mip + 1, desc.src_array,
                              desc.src_array + 1)
              .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

  reflector.addInputOutput("Target")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setSubresource(desc.dst_mip, desc.dst_mip + 1,
                                   desc.dst_array, desc.dst_array + 1)
                   .enableDepthWrite(false)
                   .setAttachmentLoc(0)
                   .setDepthCompareFn(RHI::CompareFunction::ALWAYS));

  return reflector;
}

auto BlitPass::execute(RDG::RenderContext* context,
                     RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* in = renderData.getTexture("Source");
  GFX::Texture* out = renderData.getTexture("Target");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{out->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{},
  };

  RHI::RenderPassEncoder* encoder = beginPass(context, out);

  getBindGroup(context, 0)
      ->updateBinding(std::vector<RHI::BindGroupEntry>{RHI::BindGroupEntry{
          0, RHI::BindingResource(
                 std::vector<RHI::TextureView*>{in->getSRV(0, 1, 0, 1)},
                 GFX::GFXManager::get()->samplerTable.fetch(
                     RHI::AddressMode::CLAMP_TO_EDGE, RHI::FilterMode::LINEAR,
                     RHI::MipmapFilterMode::LINEAR))}});

  dispatchFullScreen(context);

  encoder->end();
}
}  // namespace SIByL