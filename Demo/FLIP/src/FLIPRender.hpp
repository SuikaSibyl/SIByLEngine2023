#pragma once
#include <imgui.h>
#include <imgui_internal.h>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.Editor.Core.hpp>
#include <SE.GFX.hpp>
#include <SE.Math.Geometric.hpp>
#include <SE.RDG.hpp>
#include <SE.RHI.hpp>
#include <SE.SRenderer.hpp>
#include <SE.GFX-Loader.ShaderLoader.hpp>
#include <array>
#include <cmath>
#include <compare>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

namespace SIByL {
SE_EXPORT struct CellDrawPass : public RDG::FullScreenPass {
  Math::uvec2  cellSize;
  GFX::Buffer* vert_buffer;
  Core::GUID frag;
  FLIP::FLIP2DDeviceData* data = nullptr;

  CellDrawPass(Math::uvec2 cellSize) : cellSize(cellSize) {
    pConst.cellSize = cellSize;
    frag = GFX::GFXManager::get()->registerShaderModuleResource(
        "asset/cell_draw_frag.spv", {nullptr, RHI::ShaderStages::FRAGMENT});
    RDG::FullScreenPass::init(
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
  }
  virtual auto reflect() noexcept -> RDG::PassReflection {
    RDG::PassReflection reflector;
    reflector.addOutput("CellTex")
        .isTexture()
        .withSize(Math::ivec3(cellSize.x, cellSize.y, 1))
        .withFormat(RHI::TextureFormat::RGBA8_UNORM)
        .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
        .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::ColorAttachment}
                     .setAttachmentLoc(0));
    return reflector;
  }
  struct PushConstants {
    Math::uvec2 cellSize;
    float checkerboard_size = 7.0;
    uint32_t padding1;
  } pConst;
  virtual auto renderUI() noexcept -> void override {
    ImGui::DragFloat("Particle Radius", &pConst.checkerboard_size, 5);
  }
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void {
    GFX::Texture* color = renderData.getTexture("CellTex");
    renderPassDescriptor = {
        {RHI::RenderPassColorAttachment{color->getRTV(0, 0, 1),
                                        nullptr,
                                        {0, 0, 0, 1},
                                        RHI::LoadOp::CLEAR,
                                        RHI::StoreOp::STORE}},
    };
    
    std::vector<RHI::BindGroupEntry> set_0_entries;
    set_0_entries.push_back(RHI::BindGroupEntry{
        0, RHI::BindingResource(RHI::BufferBinding{
               data->cell_color_buffer, 0, data->cell_color_buffer->size()})});
    getBindGroup(context, 0)->updateBinding(set_0_entries);

    RHI::RenderPassEncoder* encoder = beginPass(context, color);
    encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::FRAGMENT, 0,
                           sizeof(PushConstants));
    encoder->draw(6, 1, 0, 0);
    encoder->end();
  }
};

SE_EXPORT struct CellQuadDrawPass : public RDG::RenderPass {
  Math::bounds2 tank_rect;

  CellQuadDrawPass(Math::bounds2 tank_rect) : tank_rect(tank_rect) {
    Math::vec2 center = (tank_rect.pMin + tank_rect.pMax) / 2;
    Math::vec2 diag = (tank_rect.pMax - tank_rect.pMin) / 2;
    pConst.bounds = Math::vec4{center, diag};

   auto [vert, frag] = GFX::ShaderLoader_SLANG::load<2u>(
        "asset/cell_quad_draw.slang",
        std::array<std::pair<std::string, RHI::ShaderStages>, 2>{
            std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
            std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
        });
    RDG::RenderPass::init(
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
  }
  virtual auto reflect() noexcept -> RDG::PassReflection {
    RDG::PassReflection reflector;
    reflector.addInput("CellTex")
        .isTexture()
        .withFormat(RHI::TextureFormat::RGBA8_UNORM)
        .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(0, 1, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
    reflector.addInputOutput("Color")
        .isTexture()
        .withSize(Math::vec3(1, 1, 1))
        .withFormat(RHI::TextureFormat::RGBA8_UNORM)
        .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
        .consume(RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::ColorAttachment}
                     .setAttachmentLoc(0));
    reflector.addInputOutput("Depth")
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
    Math::vec4 bounds;
  } pConst;
  //virtual auto renderUI() noexcept -> void override {
  //  {
  //    float radius = pConst.particle_radius;
  //    ImGui::DragFloat("Particle Radius", &radius, 0.01);
  //    pConst.particle_radius = radius;
  //  }
  //}
  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void {
    GFX::Texture* color = renderData.getTexture("Color");
    GFX::Texture* depth = renderData.getTexture("Depth");
    GFX::Texture* cellTex = renderData.getTexture("CellTex");

    renderPassDescriptor = {
        {RHI::RenderPassColorAttachment{color->getRTV(0, 0, 1),
                                        nullptr,
                                        {0, 0, 0, 1},
                                        RHI::LoadOp::LOAD,
                                        RHI::StoreOp::STORE}},
        RHI::RenderPassDepthStencilAttachment{
            depth->getDSV(0, 0, 1), 1, RHI::LoadOp::LOAD, RHI::StoreOp::STORE,
            false, 0, RHI::LoadOp::LOAD, RHI::StoreOp::DONT_CARE, false},
    };

    std::vector<RHI::BindGroupEntry> set_0_entries = {
        (*renderData.getBindGroupEntries("CommonScene"))[0]};
    GFX::Sampler* default_sampler =
        Core::ResourceManager::get()->getResource<GFX::Sampler>(
            GFX::GFXManager::get()->commonSampler.defaultSampler);
    //set_0_entries.push_back(RHI::BindGroupEntry{
    //    1, RHI::BindingResource(cellTex->getSRV(0, 1, 0, 1),
    //                            default_sampler->sampler.get())});
    //getBindGroup(context, 0)->updateBinding(set_0_entries);
    //updateBinding(context, "GlobalUniforms", set_0_entries[0].binding);
    updateBinding(context, "GlobalUniforms", set_0_entries[0].resource);
    updateBinding(
        context, "in_color",
        {cellTex->getSRV(0, 1, 0, 1), default_sampler->sampler.get()});

    RHI::RenderPassEncoder* encoder = beginPass(context, color);
    encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::VERTEX, 0,
                           sizeof(PushConstants));
    encoder->draw(6, 1, 0, 0);
    encoder->end();
  }
};

SE_EXPORT struct ParticleDrawPass : public RDG::RenderPass {
ParticleDrawPass() {
    vert = GFX::GFXManager::get()->registerShaderModuleResource(
        "asset/particle_draw_vert.spv", {nullptr, RHI::ShaderStages::VERTEX});
    frag = GFX::GFXManager::get()->registerShaderModuleResource(
        "asset/particle_draw_frag.spv",
        {nullptr, RHI::ShaderStages::FRAGMENT});
    RDG::RenderPass::init(
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

// ElasticDemo::DemoScript* script;

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
    float particle_radius = 0.f;
    uint32_t padding;
    uint32_t padding1;
    uint32_t padding2;
} pConst;

virtual auto renderUI() noexcept -> void override {
    {
    float radius = pConst.particle_radius;
    ImGui::DragFloat("Particle Radius", &radius, 0.01);
    pConst.particle_radius = radius;
    }
}

FLIP::FLIP2DDeviceData* data = nullptr;

virtual auto execute(RDG::RenderContext* context,
                        RDG::RenderData const& renderData) noexcept -> void {
    GFX::Texture* color = renderData.getTexture("Color");
    GFX::Texture* depth = renderData.getTexture("Depth");

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

     context->cmdEncoder->pipelineBarrier(RHI::BarrierDescriptor{
        (uint32_t)RHI::PipelineStages::ALL_COMMANDS_BIT,
         (uint32_t)RHI::PipelineStages::ALL_COMMANDS_BIT,
         0,
         // Optional (Memory Barriers)
         {},
         std::vector<RHI::BufferMemoryBarrierDescriptor>{
             RHI::BufferMemoryBarrierDescriptor{
                 data->particle_pos_buffer,
                 (uint32_t)RHI::AccessFlagBits::MEMORY_WRITE_BIT,
                 (uint32_t)RHI::AccessFlagBits::MEMORY_READ_BIT,
             }},
         {}});

    if (pConst.particle_radius == 0.f) {
    pConst.particle_radius = data->particle_radius;
    }
    std::vector<RHI::BindGroupEntry> set_0_entries = {
        (*renderData.getBindGroupEntries("CommonScene"))[0]};
    set_0_entries.push_back(RHI::BindGroupEntry{
        1,
        RHI::BindingResource(RHI::BufferBinding{
            data->particle_pos_buffer, 0, data->particle_pos_buffer->size()})});
    set_0_entries.push_back(RHI::BindGroupEntry{
        2,
        RHI::BindingResource(RHI::BufferBinding{
            data->particle_col_buffer, 0, data->particle_col_buffer->size()})});
    getBindGroup(context, 0)->updateBinding(set_0_entries);

    RHI::RenderPassEncoder* encoder = beginPass(context, color);
    encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::VERTEX, 0,
                            sizeof(PushConstants));
    encoder->draw(30, data->particle_count, 0, 0);
    encoder->end();
}

GFX::Buffer* vert_buffer;
Core::GUID vert, frag;
};
}  // namespace SIByL

namespace SIByL::SRP {
SE_EXPORT struct FLIPRendererGraph : public RDG::Graph {
FLIPRendererGraph(FLIP::FLIP2DDescriptor const& desc) {
    addPass(std::make_unique<CellDrawPass>(Math::uvec2{desc.cellX, desc.cellY}),
            "Cell Draw Pass");
    addPass(std::make_unique<ParticleDrawPass>(), "Particle Draw Pass");
    addPass(std::make_unique<CellQuadDrawPass>(desc.bound),
            "CellQuad Draw Pass");

    cell_pass = reinterpret_cast<CellDrawPass*>(getPass("Cell Draw Pass"));
    particle_pass = reinterpret_cast<ParticleDrawPass*>(getPass("Particle Draw Pass"));

    addEdge("Particle Draw Pass", "Color",  "CellQuad Draw Pass", "Color");
    addEdge("Particle Draw Pass", "Depth",  "CellQuad Draw Pass", "Depth");
    addEdge("Cell Draw Pass", "CellTex",    "CellQuad Draw Pass", "CellTex");

    markOutput("CellQuad Draw Pass", "Color");
  }

  CellDrawPass* cell_pass;
  ParticleDrawPass* particle_pass;
};

SE_EXPORT struct FLIPRendererPipeline : public RDG::SingleGraphPipeline {
  FLIPRendererPipeline(FLIP::FLIP2DDescriptor const& desc) : graph(desc) {
    pGraph = &graph;
  }
  FLIPRendererGraph graph;
};
}  // namespace SIByL::SRP