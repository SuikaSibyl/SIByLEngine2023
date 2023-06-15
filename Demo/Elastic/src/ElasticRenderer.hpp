#pragma once
#include <imgui.h>
#include <imgui_internal.h>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.RHI.hpp>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
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
#include <vector>
#include "ElasticDemo.hpp"

namespace SIByL {
SE_EXPORT struct ElasticObjectdPass : public RDG::RenderPass {
  ElasticObjectdPass() {
    vert = GFX::GFXManager::get()->registerShaderModuleResource(
        "asset/draw_vert.spv", {nullptr, RHI::ShaderStages::VERTEX});
    frag = GFX::GFXManager::get()->registerShaderModuleResource(
        "asset/draw_frag.spv", {nullptr, RHI::ShaderStages::FRAGMENT});
    RDG::RenderPass::init(
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
  }

  ElasticDemo::DemoScript* script;

  virtual auto reflect() noexcept -> RDG::PassReflection {
    RDG::PassReflection reflector;

    reflector.addOutput("Color")
        .isTexture()
        .withSize(Math::vec3(1, 1, 1))
        .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
        .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::ColorAttachment}
                     .setAttachmentLoc(0));

    reflector.addOutput("Depth")
        .isTexture()
        .withSize(Math::vec3(1, 1, 1))
        .withFormat(RHI::TextureFormat::DEPTH32_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT)
        .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::DepthStencilAttachment}
                     .enableDepthWrite(true)
                     .setAttachmentLoc(0)
                     .setDepthCompareFn(RHI::CompareFunction::LESS));

    return reflector;
  }

  struct PushConstants {
    uint32_t lightIndex;
    float bias = 0.005;
  } pConst;

  virtual auto renderUI() noexcept -> void override {
    {
      float bias = pConst.bias;
      ImGui::DragFloat("Bias", &bias, 0.01);
      pConst.bias = bias;
    }
  }

  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void {
    GFX::Texture* color = renderData.getTexture("Color");
    GFX::Texture* depth = renderData.getTexture("Depth");

    GFX::Texture* shadowmap = renderData.getTexture("Shadowmap");

    renderPassDescriptor = {
        {RHI::RenderPassColorAttachment{color->getRTV(0, 0, 1),
                                        nullptr,
                                        {0, 0, 0, 1},
                                        RHI::LoadOp::CLEAR,
                                        RHI::StoreOp::STORE}},
        RHI::RenderPassDepthStencilAttachment{
            depth->getDSV(0, 0, 1), 1, RHI::LoadOp::CLEAR,
            RHI::StoreOp::STORE, false, 0, RHI::LoadOp::LOAD,
            RHI::StoreOp::DONT_CARE, false},
    };

    //context->cmdEncoder->pipelineBarrier(
    //    RHI::BarrierDescriptor{(uint32_t)RHI::PipelineStages::ALL_GRAPHICS_BIT,
    //                           (uint32_t)RHI::PipelineStages::ALL_GRAPHICS_BIT,
    //                           0,
    //                           // Optional (Memory Barriers)
    //                           {},
    //                           std::vector<RHI::BufferMemoryBarrierDescriptor>{
    //                               RHI::BufferMemoryBarrierDescriptor{
    //                                     script->vertex_buffers[context->flightIdx].get(),
    //                                     (uint32_t)RHI::AccessFlagBits::MEMORY_WRITE_BIT,
    //                                     (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
    //                               }},
    //                           {}});

    RHI::RenderPassEncoder* encoder = beginPass(context, color);
    ////pConst.lightIndex = RACommon::get()->mainDirectionalLight.value().lightID;
    ////encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::FRAGMENT,
    ////                       sizeof(uint32_t), sizeof(PushConstants));
    std::vector<RHI::BindGroupEntry> set_0_entries = {
        (*renderData.getBindGroupEntries("CommonScene"))[0]};
    set_0_entries.push_back(RHI::BindGroupEntry{
        1, RHI::BindingResource(RHI::BufferBinding{
               script->vertex_buffers[context->flightIdx].get(), 0,
               script->vertex_buffers[context->flightIdx]->size()})});
    getBindGroup(context, 0)->updateBinding(set_0_entries);
    encoder->draw(script->vertex_count, 1, 0, 0);



    //renderData.getDelegate("IssueAllDrawcalls")(
    //    prepareDelegateData(context, renderData));

    encoder->end();
  }

  GFX::Buffer* vert_buffer;
  Core::GUID vert, frag;
};
}  // namespace SIByL

namespace SIByL::SRP {
SE_EXPORT struct ElasticRendererGraph : public RDG::Graph {
  ElasticRendererGraph() {
    addPass(std::make_unique<ElasticObjectdPass>(), "Elastic Object Pass");
    pass = reinterpret_cast<ElasticObjectdPass*>(getPass("Elastic Object Pass"));

    markOutput("Elastic Object Pass", "Color");
  }

  ElasticObjectdPass* pass;
};

SE_EXPORT struct ElasticRendererPipeline
    : public RDG::SingleGraphPipeline {
  ElasticRendererPipeline() { pGraph = &graph; }
  ElasticRendererGraph graph;
};
}  // namespace SIByL::SRP