#pragma once

#include <array>
#include <vector>
#include <memory>
#include <compare>
#include <typeinfo>
#include <algorithm>
#include <filesystem>
#include <functional>
#include "../../../../Application/Public/SE.Application.Config.h"
#include <SE.Editor.Core.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.SRenderer.hpp>
#include <SE.Math.Geometric.hpp>
#include <Resource/SE.Core.Resource.hpp>


namespace SIByL
{
auto fitToSceneZBound(Math::bounds3 const& aabbBounds, Math::mat4 w2l) noexcept
    -> Math::vec2;
std::vector<Math::vec4> getFrustumCornersWorldSpace(Math::mat4 const& proj,
                                                    Math::mat4 const& view);
Math::mat4 getLightSpaceMatrix(Math::mat4 const& cam_view,
                               Math::vec3 const& lightDir,
                               const float nearPlane, const float farPlane,
                               Math::bounds3 bounds,
                               Math::Transform const& w2l_transform);
std::vector<Math::mat4> getLightSpaceMatrices();

SE_EXPORT struct CascadeShadowmapOpaquePass : public RDG::RenderPass {
  CascadeShadowmapOpaquePass(uint32_t idx) : idx(idx) {
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

    reflector.addInputOutput("Depth")
        .isTexture()
        .withSize(Math::ivec3(1024, 1024, 1))
        .withLayers(4)
        .withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
        .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::DepthStencilAttachment}
                     .enableDepthWrite(true)
                     .setSubresource(0, 1, idx, idx + 1)
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
            depth->getDSV(0, idx, 1), 1, RHI::LoadOp::CLEAR,
            RHI::StoreOp::STORE, false, 0, RHI::LoadOp::CLEAR,
            RHI::StoreOp::STORE, false},
    };

    Math::mat4 mat = RACommon::get()->cascade_views[idx];
    gUni.cameraData.viewProjMat = Math::transpose(mat);
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
        RACommon::get()->structured_drawcalls.all_drawcall_device->buffer.get();

    auto const& drawcall_info =
        RACommon::get()->structured_drawcalls.opaque_drawcall;

    if (drawcall_info.drawCount != 0) {
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
    }

    encoder->end();
  }

  uint32_t idx;
  Core::GUID vert, frag;
  GFX::LightComponent* light = nullptr;
  SRenderer* srenderer = nullptr;

  SRenderer::GlobalUniforms gUni;
  GFX::StructuredUniformBufferView<SRenderer::GlobalUniforms>
      global_uniform_buffer;
};

SE_EXPORT struct CascadeShadowmapAlphaPass : public RDG::RenderPass {
  CascadeShadowmapAlphaPass(uint32_t idx) : idx(idx) {
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

  uint32_t idx;

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
                     .setSubresource(0, 1, idx, idx + 1)
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
            depth->getDSV(0, idx, 1), 1, RHI::LoadOp::LOAD, RHI::StoreOp::STORE,
            false, 0, RHI::LoadOp::LOAD, RHI::StoreOp::STORE, false},
    };

    Math::mat4 mat = RACommon::get()->cascade_views[idx];
    gUni.cameraData.viewProjMat = Math::transpose(mat);
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
        RACommon::get()->structured_drawcalls.all_drawcall_device->buffer.get();

    auto const& drawcall_info =
        RACommon::get()->structured_drawcalls.alphacut_drawcall;

    if (drawcall_info.drawCount != 0) {
      getBindGroup(context, 1)
          ->updateBinding({RHI::BindGroupEntry{
              0, RHI::BindingResource{RHI::BufferBinding{
                     indirect_draw_buffer, drawcall_info.offset,
                     sizeof(RACommon::DrawIndexedIndirectEX) *
                         drawcall_info.drawCount}}}});

      uint32_t sample_batch = renderData.getUInt("AccumIdx");
      encoder->pushConstants(&sample_batch,
                             (uint32_t)RHI::ShaderStages::FRAGMENT, 0,
                             sizeof(uint32_t));

      renderData.getDelegate("PrepareDrawcalls")(
          prepareDelegateData(context, renderData));
      encoder->drawIndexedIndirect(indirect_draw_buffer, drawcall_info.offset,
                                   drawcall_info.drawCount,
                                   sizeof(RACommon::DrawIndexedIndirectEX));
    }

    encoder->end();
  }

  Core::GUID vert, frag;
  GFX::LightComponent* light = nullptr;
  SRenderer* srenderer = nullptr;

  SRenderer::GlobalUniforms gUni;
  GFX::StructuredUniformBufferView<SRenderer::GlobalUniforms>
      global_uniform_buffer;
};

struct CascadeShadowmapDummy : public RDG::DummyPass {
  CascadeShadowmapDummy() { RDG::Pass::init(); }

  virtual auto reflect() noexcept -> RDG::PassReflection {
    RDG::PassReflection reflector;

    reflector.addOutput("Depth")
        .isTexture()
        .withSize(Math::ivec3(1024, 1024, 1))
        .withLayers(4)
        .withFormat(RHI::TextureFormat::DEPTH32_FLOAT);

    return reflector;
  }

  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void {
    std::vector<Math::mat4> mats;
    if (RACommon::get()->mainDirectionalLight.has_value()) {
      Math::mat4 transform = RACommon::get()->mainDirectionalLight->transform;
      Math::vec3 direction = Math::Transform(transform) * Math::vec3(0, 0, 1);
      direction = Math::normalize(direction);
      Math::mat4 w2l =
          Math::lookAt(Math::vec3{0}, direction, Math::vec3{0, 1, 0}).m;
      mats = getLightSpaceMatrices();
    }
    RACommon::get()->cascade_views = mats;
    {
      RACommon::get()->mainLightCSM.cascade_depths =
          RACommon::get()->cascade_distances;
      RACommon::get()->mainLightCSM.cascade_transform_0 =
          Math::transpose(mats[0]);
      RACommon::get()->mainLightCSM.cascade_transform_1 =
          Math::transpose(mats[1]);
      RACommon::get()->mainLightCSM.cascade_transform_2 =
          Math::transpose(mats[2]);
      RACommon::get()->mainLightCSM.cascade_transform_3 =
          Math::transpose(mats[3]);
      RACommon::get()->csm_info_device.setStructure(
          RACommon::get()->mainLightCSM, context->flightIdx);
    }
  }
};

SE_EXPORT struct CascadeShadowmapPass : public RDG::Subgraph {
  CascadeShadowmapPass() {}

  virtual auto alias() noexcept -> RDG::AliasDict override {
    RDG::AliasDict dict;
    dict.addAlias("Depth", CONCAT("Alpha-3"), "Depth");
    return dict;
  }

  virtual auto onRegister(RDG::Graph* graph) noexcept -> void override {
    graph->addPass(std::make_unique<CascadeShadowmapDummy>(), CONCAT("Input"));
    graph->addPass(std::make_unique<CascadeShadowmapOpaquePass>(0),
                   CONCAT("Opaque-0"));
    graph->addPass(std::make_unique<CascadeShadowmapOpaquePass>(1),
                   CONCAT("Opaque-1"));
    graph->addPass(std::make_unique<CascadeShadowmapOpaquePass>(2),
                   CONCAT("Opaque-2"));
    graph->addPass(std::make_unique<CascadeShadowmapOpaquePass>(3),
                   CONCAT("Opaque-3"));
    graph->addPass(std::make_unique<CascadeShadowmapAlphaPass>(0),
                   CONCAT("Alpha-0"));
    graph->addPass(std::make_unique<CascadeShadowmapAlphaPass>(1),
                   CONCAT("Alpha-1"));
    graph->addPass(std::make_unique<CascadeShadowmapAlphaPass>(2),
                   CONCAT("Alpha-2"));
    graph->addPass(std::make_unique<CascadeShadowmapAlphaPass>(3),
                   CONCAT("Alpha-3"));

    graph->addEdge(CONCAT("Input"), "Depth", CONCAT("Opaque-0"), "Depth");
    graph->addEdge(CONCAT("Opaque-0"), "Depth", CONCAT("Opaque-1"), "Depth");
    graph->addEdge(CONCAT("Opaque-1"), "Depth", CONCAT("Opaque-2"), "Depth");
    graph->addEdge(CONCAT("Opaque-2"), "Depth", CONCAT("Opaque-3"), "Depth");
    graph->addEdge(CONCAT("Opaque-3"), "Depth", CONCAT("Alpha-0"), "Depth");
    graph->addEdge(CONCAT("Alpha-0"), "Depth", CONCAT("Alpha-1"), "Depth");
    graph->addEdge(CONCAT("Alpha-1"), "Depth", CONCAT("Alpha-2"), "Depth");
    graph->addEdge(CONCAT("Alpha-2"), "Depth", CONCAT("Alpha-3"), "Depth");
  }
};
}