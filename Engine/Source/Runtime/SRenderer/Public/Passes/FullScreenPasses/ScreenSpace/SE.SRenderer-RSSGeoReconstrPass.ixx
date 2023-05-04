module;
#include <imgui.h>
#include <imgui_internal.h>

#include <array>
#include <compare>
#include <filesystem>
#include <functional>
#include <memory>
#include <typeinfo>

#include "../../../../../Application/Public/SE.Application.config.h"
export module SE.SRenderer.BarPass;
import SE.Platform.Window;
import SE.SRenderer;
import SE.Math.Geometric;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;
import SE.RDG;
import SE.Editor.Core;

namespace SIByL {
export struct RSSGeoReconstrPass : public RDG::ComputePass {
  uint32_t pixel_count;
  uint32_t indices_count;
  uint32_t vertices_count;
  uint32_t width, height;
  RSSGeoReconstrPass(uint32_t width, uint32_t height)
      : width(width), height(height) {
    pixel_count = width * height;
    indices_count = pixel_count * 6;
    vertices_count = pixel_count * 4;

    comp = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/compute/rssrt/"
        "rss_geo_reconstruct_comp.spv",
        {nullptr, RHI::ShaderStages::COMPUTE});
    RDG::ComputePass::init(
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
  }

  struct PushConstants {
    Math::mat4 invProjMat;
    Math::mat4 invViewMat;
    Math::uvec2 resolution;
  } pConst;

  virtual auto reflect() noexcept -> RDG::PassReflection {
    RDG::PassReflection reflector;

    reflector.addInput("Depth")
        .isTexture()
        .withFormat(RHI::TextureFormat::R32_FLOAT)
        .withUsages((uint32_t)RHI::TextureUsage::TEXTURE_BINDING)
        .consume(
            RDG::TextureInfo::ConsumeEntry{
                RDG::TextureInfo::ConsumeType::TextureBinding}
                .setSubresource(0, 1, 0, 1)
                .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

    reflector.addOutput("VertexBuffer")
        .isBuffer()
        .withSize(sizeof(Math::vec3) * vertices_count)
        .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
        .consume(
            RDG::BufferInfo::ConsumeEntry{}
                .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                           (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
                .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

    reflector.addOutput("IndicesBuffer")
        .isBuffer()
        .withSize(sizeof(uint32_t) * indices_count)
        .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
        .consume(
            RDG::BufferInfo::ConsumeEntry{}
                .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                           (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
                .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

    return reflector;
  }

  virtual auto execute(RDG::RenderContext* context,
                       RDG::RenderData const& renderData) noexcept -> void {
    GFX::Texture* depth = renderData.getTexture("Depth");
    GFX::Buffer* vb = renderData.getBuffer("VertexBuffer");
    GFX::Buffer* ib = renderData.getBuffer("IndicesBuffer");

    getBindGroup(context, 0)
        ->updateBinding(std::vector<RHI::BindGroupEntry>{
            RHI::BindGroupEntry{0, RHI::BindingResource{depth->getSRV(0, 1, 0, 1)}},
            RHI::BindGroupEntry{1, RHI::BindingResource{RHI::BufferBinding{vb->buffer.get(), 0, vb->buffer->size()}}},
            RHI::BindGroupEntry{2, RHI::BindingResource{RHI::BufferBinding{ib->buffer.get(), 0, ib->buffer->size()}}} });

    RHI::ComputePassEncoder* encoder = beginPass(context);

    pConst.resolution.x = width;
    pConst.resolution.y = height;
    SRenderer::CameraData* cd = reinterpret_cast<SRenderer::CameraData*>(renderData.getPtr("CameraData"));
    pConst.invProjMat = Math::inverse((cd->projMat));
    pConst.invViewMat = Math::inverse(cd->viewMat);

    prepareDispatch(context);

    encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                           sizeof(PushConstants));
    encoder->dispatchWorkgroups((width + 15) / 16, (height + 15) / 16, 1);

    encoder->end();
  }

  Core::GUID comp;
};
}  // namespace SIByL