#include <imgui.h>
#include <imgui_internal.h>

#include <array>
#include <compare>
#include <filesystem>
#include <functional>
#include <memory>
#include <typeinfo>

#include "../../../../../Application/Public/SE.Application.config.h"
#include <SE.Editor.Core.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include <SE.SRenderer.hpp>
#include <SE.Math.Geometric.hpp>
#include <Resource/SE.Core.Resource.hpp>

namespace SIByL {
SE_EXPORT struct RSSGeoReconstrPass : public RDG::ComputePass {
  uint32_t pixel_count;
  uint32_t indices_count;
  uint32_t vertices_count;
  uint32_t width, height;

  enum struct GeometrySetting {
    PiecewiseConstant,
    PiecewiseConstantCliff,
    Triangulate,
    TriangulateMirror,
  } geoSetting;

  auto getVerticesCount() noexcept -> uint32_t {
    switch (geoSetting) {
      case SIByL::RSSGeoReconstrPass::GeometrySetting::PiecewiseConstant:
        return width * height * 4;
      case SIByL::RSSGeoReconstrPass::GeometrySetting::PiecewiseConstantCliff:
        return width * height * 8;
      case SIByL::RSSGeoReconstrPass::GeometrySetting::Triangulate:
        return width * height * 4;
      case SIByL::RSSGeoReconstrPass::GeometrySetting::TriangulateMirror:
        return width * height * 4;
      default:
        return 0;
    }
  }

  auto getIndicesCount() noexcept -> uint32_t {
    switch (geoSetting) {
      case SIByL::RSSGeoReconstrPass::GeometrySetting::PiecewiseConstant:
        return width * height * 6;
      case SIByL::RSSGeoReconstrPass::GeometrySetting::PiecewiseConstantCliff:
        return width * height * 18;
      case SIByL::RSSGeoReconstrPass::GeometrySetting::Triangulate:
        return width * height * 6;
      case SIByL::RSSGeoReconstrPass::GeometrySetting::TriangulateMirror:
        return width * height * 6;
      default:
        return 0;
    }
  }

  RSSGeoReconstrPass(uint32_t width, uint32_t height)
      : width(width), height(height) {
    pixel_count = width * height;
    vertices_count = pixel_count * 8;
    indices_count = pixel_count * 18;

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
    int geoType;
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

    reflector.addInternal("VertexBuffer")
        .isBuffer()
        .withSize(sizeof(Math::vec3) * vertices_count)
        .withUsages(
            (uint32_t)RHI::BufferUsage::STORAGE |
            (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
            (uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
        .consume(
            RDG::BufferInfo::ConsumeEntry{}
                .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                           (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
                .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

    reflector.addInternal("IndicesBuffer")
        .isBuffer()
        .withSize(sizeof(uint32_t) * indices_count)
        .withUsages(
            (uint32_t)RHI::BufferUsage::STORAGE |
            (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
            (uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
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

    GFX::Sampler* sampler =
        Core::ResourceManager::get()->getResource<GFX::Sampler>(
            GFX::GFXManager::get()->commonSampler.defaultSampler);

    getBindGroup(context, 0)
        ->updateBinding(std::vector<RHI::BindGroupEntry>{
            RHI::BindGroupEntry{
                0,
                RHI::BindingResource{
                    depth->getSRV(0, 1, 0, 1),
                    sampler->sampler.get()}},
            RHI::BindGroupEntry{1, RHI::BindingResource{RHI::BufferBinding{vb->buffer.get(), 0, vb->buffer->size()}}},
            RHI::BindGroupEntry{2, RHI::BindingResource{RHI::BufferBinding{ib->buffer.get(), 0, ib->buffer->size()}}} });

    RHI::ComputePassEncoder* encoder = beginPass(context);

    pConst.resolution.x = width;
    pConst.resolution.y = height;
    pConst.geoType = int(geoSetting);
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