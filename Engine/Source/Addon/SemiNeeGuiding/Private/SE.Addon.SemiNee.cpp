#include "../Public/SE.Addon.SemiNee.hpp"
#include <SE.Math.ShaderCommon.hpp>
#include <addon/seminee/seminee_interface.hlsli>
#include <SE.Addon.LTC.hpp>
#include <SE.Core.Utility.hpp>

namespace SIByL::Addon::SemiNEE {

InitialSamplePass::InitialSamplePass() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "seminee/initial_sample_pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("InitialSampleRgen", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 3);
}

auto InitialSamplePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  uint32_t vplNum = 1280 * 720 * 2 - 1;
  reflector.addOutput("DiffuseVPLs")
      .isBuffer()
      .withSize(sizeof(VPLData) * vplNum)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

  return reflector;
}

auto InitialSamplePass::renderUI() noexcept -> void {
  ImGui::Checkbox("Spawn VPL", &spawnVPL);
}

auto InitialSamplePass::execute(RDG::RenderContext* context,
                                RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* vpl_buffer = renderData.getBuffer("DiffuseVPLs");

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  updateBindings(
      context,
      {{"u_VPLs", RHI::BindingResource{{vpl_buffer->buffer.get(), 0,
                                        vpl_buffer->buffer->size()}}}});

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("AccumIdx");

  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t batchIdx;
  } pConst;
  pConst.width = 1280;
  pConst.height = 720;
  pConst.batchIdx = batchIdx;

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));

  if (spawnVPL) {
    encoder->traceRays(pConst.width, pConst.height, 1);
  }

  encoder->end();
}

LeafEncodePass::LeafEncodePass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "seminee/leaf-encode.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("LeafEncodeMain", RHI::ShaderStages::COMPUTE),
      });
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto LeafEncodePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInput("DiffuseVPLs")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  reflector.addOutput("LeafCodes")
      .isBuffer()
      .withSize(sizeof(uint64_t) * 1280 * 720)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  return reflector;
}

auto LeafEncodePass::renderUI() noexcept -> void {

}

auto LeafEncodePass::execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* kb = renderData.getBuffer("LeafCodes");
  GFX::Buffer* vpl_buffer = renderData.getBuffer("DiffuseVPLs");

  updateBinding(context, "u_VPLs",
                RHI::BindingResource{
                    {vpl_buffer->buffer.get(), 0, vpl_buffer->buffer->size()}});
  updateBinding(
      context, "u_Codes",
      RHI::BindingResource{{kb->buffer.get(), 0, kb->buffer->size()}});


  struct PushConstant {
    float3 minExtent;
    uint32_t padding;
    float3 maxExtent;
    uint32_t numVPLs;
  } pConst;
  Math::bounds3* aabb =
      static_cast<Math::bounds3*>(renderData.getPtr("SceneAABB"));
  pConst.minExtent = aabb->pMin;
  pConst.maxExtent = aabb->pMax;
  pConst.numVPLs = 1280 * 720;

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((pConst.numVPLs + 255) / 256, 1, 1);
  encoder->end();
}

TreeInitPass::TreeInitPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "seminee/tree-initial-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto TreeInitPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addOutput("TreeNodes")
      .isBuffer()
      .withSize(sizeof(TreeNode) * (1280 * 720 * 2 - 1))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  
  reflector.addOutput("VPLMerges")
      .isBuffer()
      .withSize(sizeof(VPLMerge) * (1280 * 720 * 2 - 1))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  reflector.addOutput("IndirectArgs")
      .isBuffer()
      .withSize(sizeof(TreeConstrIndirectArgs) * 1)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                  (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  reflector.addInput("LeafCodesSorted")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  return reflector;
}

auto TreeInitPass::execute(RDG::RenderContext* context,
                     RDG::RenderData const& renderData) noexcept -> void {
    GFX::Buffer* tn = renderData.getBuffer("TreeNodes");
    GFX::Buffer* vm = renderData.getBuffer("VPLMerges");
    GFX::Buffer* id = renderData.getBuffer("IndirectArgs");
    GFX::Buffer* cs = renderData.getBuffer("LeafCodesSorted");

    updateBinding(
        context, "u_Nodes",
        RHI::BindingResource{{tn->buffer.get(), 0, tn->buffer->size()}});
    updateBinding(
        context, "u_Merge",
        RHI::BindingResource{{vm->buffer.get(), 0, vm->buffer->size()}});
    updateBinding(
        context, "u_ConstrIndirectArgs",
        RHI::BindingResource{{id->buffer.get(), 0, id->buffer->size()}});
    updateBinding(
        context, "u_Codes",
        RHI::BindingResource{{cs->buffer.get(), 0, cs->buffer->size()}});

    uint32_t numVPLs = 1280 * 720;
    uint32_t numTotalNodes = 1280 * 720 * 2 - 1;
    RHI::ComputePassEncoder* encoder = beginPass(context);
    encoder->pushConstants(&numVPLs, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                           sizeof(uint32_t));
    encoder->dispatchWorkgroups((numTotalNodes + 255) / 256, 1, 1);
    encoder->end();
}

TreeLeavesPass::TreeLeavesPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "seminee/tree-leaves-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto TreeLeavesPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInputOutput("TreeNodes")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  reflector.addInput("LeafCodesSorted")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  reflector.addInput("IndirectArgs")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                  (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess(
                       (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                       (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT |
                             (uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));

  return reflector;
}

auto TreeLeavesPass::execute(RDG::RenderContext* context,
                           RDG::RenderData const& renderData) noexcept -> void {
  GFX::Buffer* tn = renderData.getBuffer("TreeNodes");
  GFX::Buffer* lc = renderData.getBuffer("LeafCodesSorted");
  GFX::Buffer* ia = renderData.getBuffer("IndirectArgs");

  updateBinding(
      context, "u_Nodes",
      RHI::BindingResource{{tn->buffer.get(), 0, tn->buffer->size()}});
  updateBinding(
      context, "u_Codes",
      RHI::BindingResource{{lc->buffer.get(), 0, lc->buffer->size()}});
  updateBinding(
      context, "u_ConstrIndirectArgs",
      RHI::BindingResource{{ia->buffer.get(), 0, ia->buffer->size()}});

  uint32_t numVPLs = 1280 * 720;
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroupsIndirect(ia->buffer.get(), 0);
  encoder->end();
}

TreeInternalPass::TreeInternalPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "seminee/tree-internal-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto TreeInternalPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInputOutput("TreeNodes")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  reflector.addInput("LeafCodesSorted")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  reflector.addInput("IndirectArgs")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                  (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess(
                       (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                       (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT |
                             (uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));

  return reflector;
}

auto TreeInternalPass::execute(RDG::RenderContext* context,
                             RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* tn = renderData.getBuffer("TreeNodes");
  GFX::Buffer* lc = renderData.getBuffer("LeafCodesSorted");
  GFX::Buffer* ia = renderData.getBuffer("IndirectArgs");

  updateBinding(
      context, "u_Nodes",
      RHI::BindingResource{{tn->buffer.get(), 0, tn->buffer->size()}});
  updateBinding(
      context, "u_Codes",
      RHI::BindingResource{{lc->buffer.get(), 0, lc->buffer->size()}});
  updateBinding(
      context, "u_ConstrIndirectArgs",
      RHI::BindingResource{{ia->buffer.get(), 0, ia->buffer->size()}});

  uint32_t size = 1280 * 720;
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&size, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(uint32_t));
  encoder->dispatchWorkgroupsIndirect(ia->buffer.get(), 16);
  encoder->end();
}

TreeMergePass::TreeMergePass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "seminee/tree-merge-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto TreeMergePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInput("TreeNodes")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  reflector.addInputOutput("VPLData")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  reflector.addInputOutput("VPLMerges")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  reflector.addInput("IndirectArgs")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                  (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess(
                       (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                       (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT |
                             (uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));
 
  return reflector;
}

auto TreeMergePass::execute(RDG::RenderContext* context,
                             RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* tn = renderData.getBuffer("TreeNodes");
  GFX::Buffer* vd = renderData.getBuffer("VPLData");
  GFX::Buffer* vm = renderData.getBuffer("VPLMerges");
  GFX::Buffer* ia = renderData.getBuffer("IndirectArgs");

  updateBinding(
      context, "u_Nodes",
      RHI::BindingResource{{tn->buffer.get(), 0, tn->buffer->size()}});
  updateBinding(
      context, "u_VPLData",
      RHI::BindingResource{{vd->buffer.get(), 0, vd->buffer->size()}});
  updateBinding(
      context, "u_Merge",
      RHI::BindingResource{{vm->buffer.get(), 0, vm->buffer->size()}});
  updateBinding(
      context, "u_ConstrIndirectArgs",
      RHI::BindingResource{{ia->buffer.get(), 0, ia->buffer->size()}});

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroupsIndirect(ia->buffer.get(), 16);
  encoder->end();
}

auto TreeMergePass::renderUI() noexcept -> void {
  ImGui::DragFloat("Min Normal Score", &pConst.gApproxParams.x, 0.01f);
  ImGui::DragFloat("Max Normal ZStd", &pConst.gApproxParams.y, 0.01f);
}

TileBasedDistPass::TileBasedDistPass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "seminee/tile-based-distribution.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto TileBasedDistPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInput("VPLData")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  reflector.addOutput("VPLSelection")
      .isBuffer()
      .withSize(sizeof(int4) * 3600)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  reflector.addOutput("DebugBuffer")
      .isBuffer()
      .withSize(sizeof(float) * 3600 * 256)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  return reflector;
}

auto TileBasedDistPass::execute(RDG::RenderContext* context,
                                RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* vd = renderData.getBuffer("VPLData");
  GFX::Buffer* vs = renderData.getBuffer("VPLSelection");
  GFX::Buffer* db = renderData.getBuffer("DebugBuffer");

  updateBinding(
      context, "u_VPLData",
      RHI::BindingResource{{vd->buffer.get(), 0, vd->buffer->size()}});
  updateBinding(
      context, "u_SelectedVPL",
      RHI::BindingResource{{vs->buffer.get(), 0, vs->buffer->size()}});
  updateBinding(
      context, "u_DebugBuffer",
      RHI::BindingResource{{db->buffer.get(), 0, db->buffer->size()}});

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroups(80, 45, 1);
  encoder->end();
}

TileDistPerPixelVisPass::TileDistPerPixelVisPass() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "seminee/tile-dist-visualize-perpixel.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RaygenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 3);
}

auto TileDistPerPixelVisPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addOutput("Target")
      .isTexture()
      .withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

  reflector.addOutput("Estimate")
      .isTexture()
      .withSize(Math::ivec3(512, 512, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

    return reflector;
}

auto TileDistPerPixelVisPass::renderUI() noexcept -> void {
    ImGui::SliderInt2("Debug Pixel", &debugPixel.x, 0, 1280);
}

auto TileDistPerPixelVisPass::onInteraction(
    Platform::Input* input, Editor::Widget::WidgetInfo* info) noexcept -> void {
    if (info->isFocused && info->isHovered) {
    if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
      debugPixel.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
      debugPixel.y = std::clamp(info->mousePos.y, 0.f, 719.f);
    }
    }
}

auto TileDistPerPixelVisPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
    GFX::Texture* tt = renderData.getTexture("Target");
    GFX::Texture* et = renderData.getTexture("Estimate");

    std::vector<RHI::BindGroupEntry>* set_0_entries =
        renderData.getBindGroupEntries("CommonScene");
    getBindGroup(context, 0)->updateBinding(*set_0_entries);
    std::vector<RHI::BindGroupEntry>* set_1_entries =
        renderData.getBindGroupEntries("CommonRT");
    getBindGroup(context, 1)->updateBinding(*set_1_entries);
    
    updateBinding(context, "u_Target",
                  RHI::BindingResource{tt->getSRV(0, 1, 0, 1)});
    updateBinding(context, "u_Estimate",
                  RHI::BindingResource{et->getSRV(0, 1, 0, 1)});

    //updateBinding(
    //    context, "u_VPLData",
    //    RHI::BindingResource{{vd->buffer.get(), 0, vd->buffer->size()}});
    //updateBinding(
    //    context, "u_SelectedVPL",
    //    RHI::BindingResource{{vs->buffer.get(), 0, vs->buffer->size()}});
    //updateBinding(
    //    context, "u_DebugBuffer",
    //    RHI::BindingResource{{db->buffer.get(), 0, db->buffer->size()}});

    RHI::RayTracingPassEncoder* encoder = beginPass(context);

    uint32_t batchIdx = renderData.getUInt("AccumIdx");

    struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t batchIdx;
    int sampleHint;
    Math::ivec2 targetPixel;
    } pConst;
    pConst.width = 1280;
    pConst.height = 720;
    pConst.batchIdx = batchIdx;
    pConst.sampleHint = sample_hint;
    pConst.targetPixel = debugPixel;

    encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                           sizeof(PushConstant));
    encoder->traceRays(512, 512, 1);
    encoder->end();
}

TileDistExchangePass::TileDistExchangePass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "seminee/tile-dist-exchange.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto TileDistExchangePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInput("VPLData")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  reflector.addInputOutput("VPLSelection")
      .isBuffer()
      .withSize(sizeof(int4) * 3600)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  return reflector;
}

auto TileDistExchangePass::execute(RDG::RenderContext* context,
                                RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* vd = renderData.getBuffer("VPLData");
  GFX::Buffer* vs = renderData.getBuffer("VPLSelection");

  updateBinding(
      context, "u_VPLData",
      RHI::BindingResource{{vd->buffer.get(), 0, vd->buffer->size()}});
  updateBinding(
      context, "u_SelectedVPL",
      RHI::BindingResource{{vs->buffer.get(), 0, vs->buffer->size()}});

  RHI::ComputePassEncoder* encoder = beginPass(context);
  if (useExhange) {
    encoder->dispatchWorkgroups(16, 9, 1);
  }
  encoder->end();
}

auto TileDistExchangePass::renderUI() noexcept -> void {
  ImGui::Checkbox("Use Excchange", &useExhange);
}

TileDistSamplePass::TileDistSamplePass() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "seminee/tile-dist-sampling.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RaygenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 3);
}

auto TileDistSamplePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInput("VPLData")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

  reflector.addInput("VPLSelection")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

  reflector.addOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

  reflector.addOutput("DebugImage")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  
  return reflector;
}

auto TileDistSamplePass::renderUI() noexcept -> void {
  //ImGui::Checkbox("Spawn VPL", &spawnVPL);
  ImGui::SliderInt("sample_hint", &sample_hint, 0, 10);
}

auto TileDistSamplePass::execute(RDG::RenderContext* context,
                                RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* db = renderData.getTexture("DebugImage");
  GFX::Buffer* vpl_buffer = renderData.getBuffer("VPLData");
  GFX::Buffer* vs = renderData.getBuffer("VPLSelection");

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  updateBindings(
      context,
      {{"u_VPLData", RHI::BindingResource{{vpl_buffer->buffer.get(), 0,
                                           vpl_buffer->buffer->size()}}},
       {"u_SelectedVPL",
        RHI::BindingResource{{vs->buffer.get(), 0, vs->buffer->size()}}},
       {"u_Color", RHI::BindingResource{{color->getSRV(0, 1, 0, 1)}}},
       {"u_Debug", RHI::BindingResource{{db->getSRV(0, 1, 0, 1)}}}});

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("AccumIdx");

  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t batchIdx;
    int sampleHint;
  } pConst;
  pConst.width = 1280;
  pConst.height = 720;
  pConst.batchIdx = batchIdx;
  pConst.sampleHint = sample_hint;

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(pConst.width, pConst.height, 1);
  encoder->end();
}

TestQuadSamplePass::TestQuadSamplePass() {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "seminee/test-quad-sampling.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto TestQuadSamplePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  return reflector;
}

auto TestQuadSamplePass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* hc = Singleton<LTC::LTCCommon>::instance()->HorizonClipLUT;
  GFX::Sampler* linearSampler =
      Singleton<LTC::LTCCommon>::instance()->lutSampler;

  updateBinding(context, "CameraBuffer",
                renderData.getBindingResource("GlobalUniforms").value());
  updateBinding(context, "u_color",
                RHI::BindingResource{color->getUAV(0, 0, 1)});
  updateBinding(context, "u_HoriClipLUT",
                RHI::BindingResource{hc->getSRV(0, 1, 0, 1),
                                     linearSampler->sampler.get()});

  uint32_t batchIdx = renderData.getUInt("AccumIdx");
  pConst.sample_batch = batchIdx;

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups(1280 / 16, 720 / 16, 1);
  encoder->end();
}

auto TestQuadSamplePass::renderUI() noexcept -> void {
  ImGui::DragFloat3("Pos Emitter", &pConst.emitterPos.data[0], 0.1f);
  ImGui::DragFloat3("Dir Emitter", &pConst.emitterDirection.data[0], 0.1f);
  ImGui::DragInt("Sample Strategy", &pConst.sampleUsage, 1, 0, 5);
}
auto TestQuadSamplePass::onInteraction(
    Platform::Input* input, Editor::Widget::WidgetInfo* info) noexcept -> void {
  if (info->isFocused && info->isHovered) {
    if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
      pConst.seperator = std::clamp(info->mousePos.x, 0.f, 1279.f);
    }
  }
}
    
DVPLVisualizePass::DVPLVisualizePass() {
  auto [vert, frag] = GFX::ShaderLoader_SLANG::load<2u>(
      "../Engine/Shaders/SRenderer/addon/"
      "seminee/dvpl-visualize.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 2>{
          std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto DVPLVisualizePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInput("DiffuseVPLs")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT |
                        (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

  reflector.addOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA8_UNORM)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(0));

  reflector.addInternal("Depth")
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

auto DVPLVisualizePass::execute(RDG::RenderContext* context,
                                RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* depth = renderData.getTexture("Depth");
  GFX::Buffer* vplBuffer = renderData.getBuffer("DiffuseVPLs");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{color->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{
          depth->getDSV(0, 0, 1), 1, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE,
          false, 0, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE, false},
  };

  updateBinding(context, "CameraBuffer",
                renderData.getBindingResource("GlobalUniforms").value());
  updateBinding(context, "u_VPLs",
                RHI::BindingResource{
                    {vplBuffer->buffer.get(), 0, vplBuffer->buffer->size()}});

  RHI::RenderPassEncoder* encoder = beginPass(context, color);

  pConst.vplIdOffset = 0;

  encoder->pushConstants(&pConst,
                         (uint32_t)RHI::ShaderStages::VERTEX |
                             (uint32_t)RHI::ShaderStages::FRAGMENT,
                         0, sizeof(PushConstantBuffer));

  encoder->draw(30, 1280 * 720, 0, 0);

  encoder->end();
}

auto DVPLVisualizePass::renderUI() noexcept -> void {
  ImGui::DragFloat("Color scale", &pConst.gVPLColorScale, 0.01);
  ImGui::DragFloat("Size scale", &pConst.gVPLRenderScale, 0.01);
}

TileDistVisualizePass::TileDistVisualizePass() {
  auto [vert, frag] = GFX::ShaderLoader_SLANG::load<2u>(
      "../Engine/Shaders/SRenderer/addon/"
      "seminee/tile-dist-visualize.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 2>{
          std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

TreeVisualizePass::TreeVisualizePass() {
  auto [vert, frag] = GFX::ShaderLoader_SLANG::load<2u>(
      "../Engine/Shaders/SRenderer/addon/"
      "seminee/tree-visualize.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 2>{
          std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto TileDistVisualizePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInput("VPLData")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT |
                        (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

  reflector.addInput("SelectedVPL")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT));


  reflector.addOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA8_UNORM)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(0));

  reflector.addInternal("Depth")
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

auto TileDistVisualizePass::execute(RDG::RenderContext* context,
                                RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* depth = renderData.getTexture("Depth");
  GFX::Buffer* vplBuffer = renderData.getBuffer("VPLData");
  GFX::Buffer* sv = renderData.getBuffer("SelectedVPL");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{color->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{
          depth->getDSV(0, 0, 1), 1, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE,
          false, 0, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE, false},
  };

  updateBinding(context, "CameraBuffer",
                renderData.getBindingResource("GlobalUniforms").value());
  updateBinding(context, "u_VPLData",
                RHI::BindingResource{
                    {vplBuffer->buffer.get(), 0, vplBuffer->buffer->size()}});
  updateBinding(
      context, "u_SelectedVPL",
      RHI::BindingResource{{sv->buffer.get(), 0, sv->buffer->size()}});

  RHI::RenderPassEncoder* encoder = beginPass(context, color);

  encoder->pushConstants(&pConst,
                         (uint32_t)RHI::ShaderStages::VERTEX |
                             (uint32_t)RHI::ShaderStages::FRAGMENT,
                         0, sizeof(PushConstantBuffer));
  encoder->draw(6, 4);
  // encoder->draw(30, 1280 * 720, 0, 0);
  encoder->end();
}

auto TileDistVisualizePass::renderUI() noexcept -> void {
  ImGui::SliderInt("Pixel X", &pConst.selectedPixel.x, 0, 1279);
  ImGui::SliderInt("Pixel Y", &pConst.selectedPixel.y, 0, 719);
}

auto TileDistVisualizePass::onInteraction(
    Platform::Input* input, Editor::Widget::WidgetInfo* info) noexcept -> void {
  if (info->isFocused && info->isHovered) {
    if (input->isMouseButtonPressed(Platform::SIByL_MOUSE_BUTTON_1)) {
      pConst.selectedPixel.x = std::clamp(info->mousePos.x, 0.f, 1279.f);
      pConst.selectedPixel.y = std::clamp(info->mousePos.y, 0.f, 719.f);
    }
  }
}

auto TreeVisualizePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addInput("DiffuseVPLs")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT |
                        (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));

  reflector.addInput("TreeNodes")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT));

  reflector.addInput("IndirectArgs")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                  (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess(
                       (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                       (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT |
                             (uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));

  reflector.addOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA8_UNORM)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(0));

  reflector.addInternal("Depth")
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

auto TreeVisualizePass::execute(RDG::RenderContext* context,
                                RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* depth = renderData.getTexture("Depth");
  GFX::Buffer* vplBuffer = renderData.getBuffer("DiffuseVPLs");
  GFX::Buffer* tn = renderData.getBuffer("TreeNodes");
  GFX::Buffer* ia = renderData.getBuffer("IndirectArgs");

  renderPassDescriptor = {
      {RHI::RenderPassColorAttachment{color->getRTV(0, 0, 1),
                                      nullptr,
                                      {0, 0, 0, 1},
                                      RHI::LoadOp::CLEAR,
                                      RHI::StoreOp::STORE}},
      RHI::RenderPassDepthStencilAttachment{
          depth->getDSV(0, 0, 1), 1, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE,
          false, 0, RHI::LoadOp::CLEAR, RHI::StoreOp::STORE, false},
  };

  updateBinding(context, "CameraBuffer",
                renderData.getBindingResource("GlobalUniforms").value());
  updateBinding(context, "u_VPLData",
                RHI::BindingResource{
                    {vplBuffer->buffer.get(), 0, vplBuffer->buffer->size()}});
  updateBinding(
      context, "u_Nodes",
      RHI::BindingResource{{tn->buffer.get(), 0, tn->buffer->size()}});
  updateBinding(
      context, "u_ConstrIndirectArgs",
      RHI::BindingResource{{ia->buffer.get(), 0, ia->buffer->size()}});

  RHI::RenderPassEncoder* encoder = beginPass(context, color);

  pConst.vplIdOffset = 0;

  encoder->pushConstants(&pConst,
                         (uint32_t)RHI::ShaderStages::VERTEX |
                             (uint32_t)RHI::ShaderStages::FRAGMENT,
                         0, sizeof(PushConstantBuffer));
  encoder->drawIndirect(ia->buffer.get(), 48, 1, 16);
  //encoder->draw(30, 1280 * 720, 0, 0);
  encoder->end();
}

auto TreeVisualizePass::renderUI() noexcept -> void {
  //ImGui::Checkbox("Show Leaves", )

  ImGui::DragFloat("Color scale", &pConst.gVPLColorScale, 0.01);
  ImGui::DragFloat("Size scale", &pConst.gVPLRenderScale, 0.01);
}

GroundTruthPass::GroundTruthPass() {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "seminee/groundtruth-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("UDPTRgen", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 3);
}

auto GroundTruthPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;

  reflector.addOutput("Color")
      .isTexture()
      .withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage(
                  (uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));

  return reflector;
}

auto GroundTruthPass::renderUI() noexcept -> void {
    {
    const char* item_names[] = {"DI Only", "GI Only", "Combined", "Multi"};
    ImGui::Combo("Mode", &showMode, item_names, IM_ARRAYSIZE(item_names),
                 IM_ARRAYSIZE(item_names));
    ImGui::DragInt("Extra bounce", &extra_bounce, 1);
    ImGui::DragInt("SPP", &spp, 1);
    ImGui::DragInt("SPP One-bounce", &spp_ob, 1);
    }
}

auto GroundTruthPass::execute(RDG::RenderContext* context,
                     RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* color = renderData.getTexture("Color");

  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);
  getBindGroup(context, 2)
      ->updateBinding({RHI::BindGroupEntry{
          0, RHI::BindingResource{color->getUAV(0, 0, 1)}}});

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  uint32_t batchIdx = renderData.getUInt("FrameIdx");

  struct PushConstant {
    uint32_t width;
    uint32_t height;
    uint32_t sample_batch;
    int showMode;
    int extra_bounce;
    int spp;
    int spp_ob;
  };
  PushConstant pConst = {color->texture->width(), color->texture->height(),
                         batchIdx, showMode, extra_bounce, spp, spp_ob};
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(color->texture->width(), color->texture->height(), 1);

  encoder->end();
}
}