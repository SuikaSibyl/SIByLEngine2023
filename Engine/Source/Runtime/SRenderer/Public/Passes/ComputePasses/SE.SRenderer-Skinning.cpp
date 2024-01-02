#include "SE.SRenderer-Skinning.hpp"

namespace SIByL {
SkinningPass::SkinningPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
    "../Engine/Shaders/SRenderer/compute/skeleton/skinning.slang",
    std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
      std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),});
  RDG::ComputePass::init(
    Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto SkinningPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  return reflector;
}

auto SkinningPass::execute(RDG::RenderContext* context,
  RDG::RenderData const& renderData) noexcept -> void {
  getBindGroup(context, 0)->updateBinding(entries);
  RHI::ComputePassEncoder* encoder = beginPass(context);
  Math::uvec2 pconst = {vertex_max, joint_max};
  encoder->pushConstants(&pconst, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(pconst));
  encoder->dispatchWorkgroups((vertex_max + 511) / 512, 1, 1);
  encoder->end();
}
}  // namespace SIByL