#include "../Public/SE.Addon.PSFiltering.hpp"

namespace SIByL::Addon::PSFiltering {
HashFunctionViewerPass::HashFunctionViewerPass() {
  auto [frag] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/pathspacefiltering/hashfunc-viewer.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),}
  );
  RDG::FullScreenPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto HashFunctionViewerPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("Color")
      .isTexture().withSize(Math::ivec3(960, 540, 1))
      .withFormat(RHI::TextureFormat::RGBA8_UNORM)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
            .setAttachmentLoc(0));
  return reflector;
}

auto HashFunctionViewerPass::execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* out = renderData.getTexture("Color");
  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{out->getRTV(0, 0, 1),
        nullptr, {0, 0, 0, 1}, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{},
  };

  struct PushConstant {
    Math::ivec2 iResolution;
    float iTime;
  } pConst = {{960, 540},
              ((Core::Timer*)renderData.getPtr("Timer"))->totalTime()};

  RHI::RenderPassEncoder* encoder = beginPass(context, out);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::FRAGMENT, 0, sizeof(PushConstant));
  dispatchFullScreen(context);
  encoder->end();
}

auto HashFunctionViewerPass::renderUI() noexcept -> void {
  const char* item_names[] = {
      "pcg3d", 
  };
  ImGui::Combo("Hash Function", &flag, item_names, IM_ARRAYSIZE(item_names), IM_ARRAYSIZE(item_names));
}
}
