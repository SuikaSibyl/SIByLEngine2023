#include "../Public/SE.Addon.Fluid.hpp"

namespace SIByL::Addon::Fluid {
LBMD2Q9Pass::LBMD2Q9Pass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
    "../Engine/Shaders/SRenderer/addon/fluidsim/lbm2d.slang",
    std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
        std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE), });
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto LBMD2Q9Pass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("Uxy")
    .isTexture().withSize(Math::ivec3(400, 100, 1))
    .withFormat(RHI::TextureFormat::RG32_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
      RDG::TextureInfo::ConsumeType::StorageBinding}
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Cylinder")
    .isTexture().withSize(Math::ivec3(400, 100, 1))
    .withFormat(RHI::TextureFormat::R32_UINT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
      RDG::TextureInfo::ConsumeType::StorageBinding}
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Lattice")
    .isBuffer().withSize(400 * 100 * sizeof(float) * 9)
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto LBMD2Q9Pass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* uxy = renderData.getTexture("Uxy");
  GFX::Texture* cylinder = renderData.getTexture("Cylinder");
  GFX::Buffer* lattice = renderData.getBuffer("Lattice");

  updateBindings(context, {
    {"u_uxy", RHI::BindingResource{uxy->getUAV(0, 0, 1)}},
    {"u_cylinder", RHI::BindingResource{cylinder->getUAV(0, 0, 1)}},
    {"u_lattice", RHI::BindingResource{{lattice->buffer.get(), 0, lattice->buffer->size()}}},
  });

  static bool initialized = false;
  struct PushConstant {
    int clearAll;
  } pConst = { initialized ? 0 : 1 };
  initialized = true;

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(PushConstant));
  encoder->dispatchWorkgroups((400 + 15) / 16, (100 + 15) / 16, 1);
  encoder->end();
}

auto LBMD2Q9Pass::renderUI() noexcept -> void {
  //ImGui::Checkbox("Clear", &clear);
}

LBMVisPass::LBMVisPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
    "../Engine/Shaders/SRenderer/addon/fluidsim/lbm2d-vis.slang",
    std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
        std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE), });
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto LBMVisPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("Color")
    .isTexture().withSize(Math::ivec3(400, 100, 1))
    .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
    .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
      RDG::TextureInfo::ConsumeType::StorageBinding}
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("Uxy")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
      RDG::TextureInfo::ConsumeType::StorageBinding}
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("Cylinder")
    .isTexture().withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
    .consume(RDG::TextureInfo::ConsumeEntry{
      RDG::TextureInfo::ConsumeType::StorageBinding}
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto LBMVisPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* cylinder = renderData.getTexture("Cylinder");
  GFX::Texture* uxy = renderData.getTexture("Uxy");

  updateBindings(context, {
    {"u_color", RHI::BindingResource{color->getUAV(0, 0, 1)}},
    {"u_cylinder", RHI::BindingResource{cylinder->getUAV(0, 0, 1)}},
    {"u_uxy", RHI::BindingResource{uxy->getUAV(0, 0, 1)}},
  });

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroups((400 + 15) / 16, (100 + 15) / 16, 1);
  encoder->end();
}

LBMGraph::LBMGraph() {
  // Create VBuffer + GBuffer, as well as history GBuffer, A-SVGF buffers
  addPass(std::make_unique<LBMD2Q9Pass>(), "LBMD2Q9 Pass");
  addPass(std::make_unique<LBMVisPass>(), "LBMVis Pass");
  addEdge("LBMD2Q9 Pass", "Uxy", "LBMVis Pass", "Uxy");
  addEdge("LBMD2Q9 Pass", "Cylinder", "LBMVis Pass", "Cylinder");
  markOutput("LBMVis Pass", "Color");
}
}