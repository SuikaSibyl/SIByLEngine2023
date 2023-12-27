#include "SE.Addon.Lightmap.hpp"
#include <xatlas.h>

namespace SIByL::Addon::Lightmap {
RasterizedLightmapUVPass::RasterizedLightmapUVPass() {
  auto [vert, frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/lightmap/"
      "lightmap-uv-viewer.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 2>{
          std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RDG::RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto RasterizedLightmapUVPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("UVMap")
      .isTexture().withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::RGBA8_UNORM)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
            .setAttachmentLoc(0));
  return reflector;
}

auto RasterizedLightmapUVPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* uv_map = renderData.getTexture("UVMap");
  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{uv_map->getRTV(0, 0, 1),
        nullptr, {0, 0, 0, 1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{},
  };

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  updateBindings(context, {
    {"UV2Buffer", renderData.getBindingResource("UV2Buffer").value()},
  });

  RHI::RenderPassEncoder* encoder = beginPass(context, uv_map);
  renderData.getDelegate("IssueAllDrawcalls")(
      prepareDelegateData(context, renderData));
  encoder->end();
}

LightmapVisualizeGraph::LightmapVisualizeGraph() {
  addPass(std::make_unique<RasterizedLightmapUVPass>(), "LightmapUV Pass");
  markOutput("LightmapUV Pass", "UVMap");
}
}