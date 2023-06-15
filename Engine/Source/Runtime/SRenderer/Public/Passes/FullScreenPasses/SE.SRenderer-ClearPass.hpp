#pragma once

#include <Resource/SE.Core.Resource.hpp>
#include <SE.Editor.Core.hpp>
#include <SE.GFX.hpp>
#include <SE.Math.Geometric.hpp>
#include <SE.RDG.hpp>
#include <SE.RHI.hpp>
#include <SE.SRenderer.hpp>
#include <array>
#include <compare>
#include <filesystem>
#include <functional>
#include <memory>
#include <typeinfo>

#include "../../../../Application/Public/SE.Application.Config.h"

namespace SIByL {
SE_EXPORT struct ClearPassR32f : public RDG::FullScreenPass {
  ClearPassR32f() {
    frag = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/"
        "clear_f32_frag.spv",
        {nullptr, RHI::ShaderStages::FRAGMENT});
    RDG::FullScreenPass::init(
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
  }

  virtual auto reflect() noexcept -> RDG::PassReflection {
    RDG::PassReflection reflector;

    reflector.addInputOutput("Target")
        .isTexture()
        .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
        .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::ColorAttachment}
                     .enableDepthWrite(false)
                     .setAttachmentLoc(0)
                     .setDepthCompareFn(RHI::CompareFunction::ALWAYS));

    return reflector;
  }

  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void {
    GFX::Texture* target = renderData.getTexture("Target");

    renderPassDescriptor = {
        {RHI::RenderPassColorAttachment{target->getRTV(0, 0, 1),
                                        nullptr,
                                        {0, 0, 0, 1},
                                        RHI::LoadOp::CLEAR,
                                        RHI::StoreOp::STORE}},
        RHI::RenderPassDepthStencilAttachment{},
    };

    RHI::RenderPassEncoder* encoder = beginPass(context, target);

    dispatchFullScreen(context);

    encoder->end();
  }

  Core::GUID vert, frag;
};
}  // namespace SIByL