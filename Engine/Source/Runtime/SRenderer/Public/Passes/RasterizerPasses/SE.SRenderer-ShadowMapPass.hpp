#pragma once

#include <array>
#include <memory>
#include <compare>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include "../../../../Application/Public/SE.Application.Config.h"
#include <SE.Math.Geometric.hpp>
#include <SE.Editor.Core.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.SRenderer.hpp>

namespace SIByL
{
SE_EXPORT auto fitToScene(Math::bounds3 const& aabbBounds,
                          Math::mat4 w2l) noexcept -> Math::mat4;

SE_EXPORT struct ShadowmapOpaquePass : public RDG::RenderPass {
  ShadowmapOpaquePass() {
    vert = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/prez_pass/"
        "prez_pass_indirect_vert.spv",
        {nullptr, RHI::ShaderStages::VERTEX});
    frag = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/prez_pass/"
        "prez_pass_frag.spv",
        {nullptr, RHI::ShaderStages::FRAGMENT});
    RDG::RenderPass::init(
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));

    global_uniform_buffer =
        GFX::GFXManager::get()
            ->createStructuredUniformBuffer<SRenderer::GlobalUniforms>();
  }

  virtual auto reflect() noexcept -> RDG::PassReflection {
    RDG::PassReflection reflector;

    reflector.addOutput("Depth")
        .isTexture()
        .withSize(Math::ivec3(1024, 1024, 1))
        .withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
        .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::DepthStencilAttachment}
                     .enableDepthWrite(true)
                     .setAttachmentLoc(0)
                     .setDepthCompareFn(RHI::CompareFunction::LESS));

    return reflector;
  }

  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void {
    GFX::Texture* depth = renderData.getTexture("Depth");

    renderPassDescriptor = {
        {},
        RHI::RenderPassDepthStencilAttachment{
            depth->getDSV(0, 0, 1), 1, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE,
            false, 0, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE, false},
    };

    if (RACommon::get()->mainDirectionalLight.has_value()) {
      Math::mat4 transform = RACommon::get()->mainDirectionalLight->transform;
      Math::vec3 direction = Math::Transform(transform) * Math::vec3(0, 0, 1);
      direction = Math::normalize(direction);
      Math::mat4 w2l =
          Math::lookAt(Math::vec3{0}, direction, Math::vec3{0, 1, 0}).m;
      Math::mat4 proj = fitToScene(RACommon::get()->sceneAABB, w2l);
      gUni.cameraData.viewMat = Math::transpose(w2l);
      gUni.cameraData.viewProjMat =
          gUni.cameraData.viewMat * Math::transpose(proj);

      RACommon::get()->shadowmapData.resize(1);
      RACommon::get()->shadowmapData[0] =
          RACommon::ShadowmapInfo{gUni.cameraData.viewProjMat};
    }
    global_uniform_buffer.setStructure(gUni, context->flightIdx);

    RHI::RenderPassEncoder* encoder = beginPass(context, depth);

    std::vector<RHI::BindGroupEntry> set_0_entries =
        *renderData.getBindGroupEntries("CommonScene");
    set_0_entries[0] = RHI::BindGroupEntry{
        0, RHI::BindingResource{
               global_uniform_buffer.getBufferBinding(context->flightIdx)}};
    getBindGroup(context, 0)->updateBinding(set_0_entries);

    // renderData.getDelegate("IssueAllDrawcalls")(prepareDelegateData(context,
    // renderData));

    RHI::Buffer* indirect_draw_buffer =
        RACommon::get()->structured_drawcalls.all_drawcalls.get_primal();

    auto const& drawcall_info =
        RACommon::get()->structured_drawcalls.opaque_drawcall;

    getBindGroup(context, 1)
        ->updateBinding({RHI::BindGroupEntry{
            0, RHI::BindingResource{RHI::BufferBinding{
                   indirect_draw_buffer, drawcall_info.offset,
                   sizeof(RACommon::DrawIndexedIndirectEX) *
                       drawcall_info.drawCount}}}});

    renderData.getDelegate("PrepareDrawcalls")(
        prepareDelegateData(context, renderData));
    encoder->drawIndexedIndirect(indirect_draw_buffer, drawcall_info.offset,
                                 drawcall_info.drawCount,
                                 sizeof(RACommon::DrawIndexedIndirectEX));

    encoder->end();
  }

  Core::GUID vert, frag;
  GFX::LightComponent* light = nullptr;
  SRenderer* srenderer = nullptr;

  SRenderer::GlobalUniforms gUni;
  GFX::StructuredUniformBufferView<SRenderer::GlobalUniforms>
      global_uniform_buffer;
};

SE_EXPORT struct ShadowmapAlphaPass : public RDG::RenderPass {
  ShadowmapAlphaPass() {
    vert = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/prez_pass/"
        "prez_pass_alpha_indirect_vert.spv",
        {nullptr, RHI::ShaderStages::VERTEX});
    frag = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/prez_pass/"
        "prez_pass_alpha_indirect_frag.spv",
        {nullptr, RHI::ShaderStages::FRAGMENT});
    RDG::RenderPass::init(
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));

    global_uniform_buffer =
        GFX::GFXManager::get()
            ->createStructuredUniformBuffer<SRenderer::GlobalUniforms>();
  }

  virtual auto reflect() noexcept -> RDG::PassReflection {
    RDG::PassReflection reflector;

    reflector.addInputOutput("Depth")
        .isTexture()
        .withSize(Math::ivec3(1024, 1024, 1))
        .withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
        .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::DepthStencilAttachment}
                     .enableDepthWrite(true)
                     .setAttachmentLoc(0)
                     .setDepthCompareFn(RHI::CompareFunction::LESS));

    return reflector;
  }

  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void {
    GFX::Texture* depth = renderData.getTexture("Depth");

    renderPassDescriptor = {
        {},
        RHI::RenderPassDepthStencilAttachment{
            depth->getDSV(0, 0, 1), 1, RHI::LoadOp::LOAD, RHI::StoreOp::STORE,
            false, 0, RHI::LoadOp::LOAD, RHI::StoreOp::STORE, false},
    };

    if (RACommon::get()->mainDirectionalLight.has_value()) {
      Math::mat4 transform = RACommon::get()->mainDirectionalLight->transform;
      Math::vec3 direction = Math::Transform(transform) * Math::vec3(0, 0, 1);
      direction = Math::normalize(direction);
      Math::mat4 w2l =
          Math::lookAt(Math::vec3{0}, direction, Math::vec3{0, 1, 0}).m;
      Math::mat4 proj = fitToScene(RACommon::get()->sceneAABB, w2l);
      gUni.cameraData.viewMat = Math::transpose(w2l);
      gUni.cameraData.viewProjMat =
          gUni.cameraData.viewMat * Math::transpose(proj);

      RACommon::get()->shadowmapData.resize(1);
      RACommon::get()->shadowmapData[0] =
          RACommon::ShadowmapInfo{gUni.cameraData.viewProjMat};
    }
    global_uniform_buffer.setStructure(gUni, context->flightIdx);

    RHI::RenderPassEncoder* encoder = beginPass(context, depth);

    std::vector<RHI::BindGroupEntry> set_0_entries =
        *renderData.getBindGroupEntries("CommonScene");
    set_0_entries[0] = RHI::BindGroupEntry{
        0, RHI::BindingResource{
               global_uniform_buffer.getBufferBinding(context->flightIdx)}};
    getBindGroup(context, 0)->updateBinding(set_0_entries);

    // renderData.getDelegate("IssueAllDrawcalls")(prepareDelegateData(context,
    // renderData));

    RHI::Buffer* indirect_draw_buffer =
        RACommon::get()->structured_drawcalls.all_drawcalls.get_primal();

    auto const& drawcall_info =
        RACommon::get()->structured_drawcalls.alphacut_drawcall;

    getBindGroup(context, 1)
        ->updateBinding({RHI::BindGroupEntry{
            0, RHI::BindingResource{RHI::BufferBinding{
                   indirect_draw_buffer, drawcall_info.offset,
                   sizeof(RACommon::DrawIndexedIndirectEX) *
                       drawcall_info.drawCount}}}});

    uint32_t sample_batch = renderData.getUInt("AccumIdx");
    encoder->pushConstants(&sample_batch, (uint32_t)RHI::ShaderStages::FRAGMENT,
                           0, sizeof(uint32_t));

    renderData.getDelegate("PrepareDrawcalls")(
        prepareDelegateData(context, renderData));
    encoder->drawIndexedIndirect(indirect_draw_buffer, drawcall_info.offset,
                                 drawcall_info.drawCount,
                                 sizeof(RACommon::DrawIndexedIndirectEX));

    encoder->end();
  }

  Core::GUID vert, frag;
  GFX::LightComponent* light = nullptr;
  SRenderer* srenderer = nullptr;

  SRenderer::GlobalUniforms gUni;
  GFX::StructuredUniformBufferView<SRenderer::GlobalUniforms>
      global_uniform_buffer;
};

SE_EXPORT struct ShadowmapPass : public RDG::Subgraph {
  ShadowmapPass() {}

  virtual auto alias() noexcept -> RDG::AliasDict override {
    RDG::AliasDict dict;
    dict.addAlias("Depth", CONCAT("Alpha"), "Depth");
    return dict;
  }

  virtual auto onRegister(RDG::Graph* graph) noexcept -> void override {
    graph->addPass(std::make_unique<ShadowmapOpaquePass>(), CONCAT("Opaque"));
    graph->addPass(std::make_unique<ShadowmapAlphaPass>(), CONCAT("Alpha"));

    graph->addEdge(CONCAT("Opaque"), "Depth", CONCAT("Alpha"), "Depth");
  }
};
}