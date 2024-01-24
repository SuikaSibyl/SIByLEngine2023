#include "../Public/SE.Addon.SST.hpp"
#include "../Public/SE.Addon.VBuffer.hpp"
#include <SE.Addon.Postprocess.hpp>

namespace SIByL::Addon::SST {
TreeEncodePass::TreeEncodePass(VPL::VPLSpawnInfo* info) : vplInfo(info) {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "sst/tree-encode.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

struct TreeConstrIndirectArgs {
  Math::ivec3 dispatch_leaf;
  uint32_t numValidVPLs;
  Math::ivec3 dispatch_internal;
  uint32_t padding0;
  Math::ivec3 dispatch_node;
  uint32_t padding1;
  Math::ivec4 draw_rects;
};

struct VPLData {
  Math::vec3 posW;
  float aabbMax_x;
  Math::vec3 normW;
  float aabbMax_y;
  Math::vec3 color;
  float intensity;
  // 16
  Math::vec3 aabbMin;
  float aabbMax_z;
  // 16
  Math::vec3 variance;
  float earlyStop;
  // 16
  int id;
  int idChild1;
  int idChild2;
  int numVPLSubTree;
};

auto TreeEncodePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("LeafCodes")
      .isBuffer().withSize(sizeof(uint64_t) * vplInfo->maxNumber * 2)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("VPLData")
      .isBuffer()
      .withSize(sizeof(VPLData) * vplInfo->maxNumber * 2)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("VPLPositions")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("VPLNormals")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("VPLColors")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("CounterBuffer")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("IndirectArgs")
      .isBuffer().withSize(sizeof(TreeConstrIndirectArgs) * 1)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                  (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));

  return reflector;
}

auto TreeEncodePass::execute(RDG::RenderContext* context,
                             RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* codes = renderData.getBuffer("LeafCodes");
  GFX::Buffer* poses = renderData.getBuffer("VPLPositions");
  GFX::Buffer* norms = renderData.getBuffer("VPLNormals");
  GFX::Buffer* color = renderData.getBuffer("VPLColors");
  GFX::Buffer* count = renderData.getBuffer("CounterBuffer");
  GFX::Buffer* indir = renderData.getBuffer("IndirectArgs");
  GFX::Buffer* vpldt = renderData.getBuffer("VPLData");

  updateBindings(context, {
    {"u_vplPositions", RHI::BindingResource{{poses->buffer.get(), 0, poses->buffer->size()}}},
    {"u_vplNormals", RHI::BindingResource{{norms->buffer.get(), 0, norms->buffer->size()}}},
    {"u_vplCounter", RHI::BindingResource{{count->buffer.get(), 0, count->buffer->size()}}},
    {"u_vplColors", RHI::BindingResource{{color->buffer.get(), 0, color->buffer->size()}}},
    {"u_Codes", RHI::BindingResource{{codes->buffer.get(), 0, codes->buffer->size()}}},
    {"u_VPLData", RHI::BindingResource{{vpldt->buffer.get(), 0, vpldt->buffer->size()}}},
    {"u_ConstrIndirectArgs", RHI::BindingResource{{indir->buffer.get(), 0, indir->buffer->size()}}},
  });
  
  struct PushConstant {
    Math::vec3 minExtent;
    uint32_t padding0;
    Math::vec3 maxExtent;
    uint32_t padding1;
  } pConst;
  Math::bounds3* aabb =
      static_cast<Math::bounds3*>(renderData.getPtr("SceneAABB"));
  pConst.minExtent = aabb->pMin;
  pConst.maxExtent = aabb->pMax;

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((vplInfo->maxNumber + 255) / 256, 1, 1);
  encoder->end();
}

struct TreeNode {
  uint32_t parent_idx;  // parent node
  uint32_t left_idx;    // index of left  child node
  uint32_t right_idx;   // index of right child node
  uint32_t vpl_idx;     // == 0xFFFFFFFF if internal node.
  uint32_t flag;        // got node already processed?
};

TreeInitPass::TreeInitPass(VPL::VPLSpawnInfo* info) : vplInfo(info) {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "sst/tree-initial-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto TreeInitPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("TreeNodes")
      .isBuffer().withSize(sizeof(TreeNode) * (vplInfo->maxNumber * 2 - 1))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("VPLMerges")
      .isBuffer().withSize(sizeof(Math::vec2) * (vplInfo->maxNumber * 2 - 1))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("IndirectArgs")
      .isBuffer().withSize(sizeof(TreeConstrIndirectArgs) * 1)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                  (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                              (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT |
                             (uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));
  return reflector;
}

auto TreeInitPass::execute(RDG::RenderContext* context,
                             RDG::RenderData const& renderData) noexcept -> void {
  GFX::Buffer* nodes = renderData.getBuffer("TreeNodes");
  GFX::Buffer* mergs = renderData.getBuffer("VPLMerges");
  GFX::Buffer* indir = renderData.getBuffer("IndirectArgs");

  updateBindings(context, {
    {"u_Nodes", RHI::BindingResource{{nodes->buffer.get(), 0, nodes->buffer->size()}}},
    {"u_Merge", RHI::BindingResource{{mergs->buffer.get(), 0, mergs->buffer->size()}}},
    {"u_ConstrIndirectArgs", RHI::BindingResource{{indir->buffer.get(), 0, indir->buffer->size()}}},
  });
  
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroupsIndirect(indir->buffer.get(), sizeof(uint32_t) * 8);
  encoder->end();
}

TreeAssignLeafIndex::TreeAssignLeafIndex(VPL::VPLSpawnInfo* info) : vplInfo(info) {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "sst/tree-leaves-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto TreeAssignLeafIndex::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("TreeNodes")
      .isBuffer().withSize(sizeof(TreeNode) * (vplInfo->maxNumber * 2 - 1))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("SortedCodes")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("IndirectArgs")
      .isBuffer().withSize(sizeof(TreeConstrIndirectArgs) * 1)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                  (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                              (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT |
                             (uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));
  return reflector;
}

auto TreeAssignLeafIndex::execute(RDG::RenderContext* context,
                             RDG::RenderData const& renderData) noexcept -> void {
  GFX::Buffer* nodes = renderData.getBuffer("TreeNodes");
  GFX::Buffer* codes = renderData.getBuffer("SortedCodes");
  GFX::Buffer* indir = renderData.getBuffer("IndirectArgs");

  updateBindings(context, {
    {"u_Nodes", RHI::BindingResource{{nodes->buffer.get(), 0, nodes->buffer->size()}}},
    {"u_Codes", RHI::BindingResource{{codes->buffer.get(), 0, codes->buffer->size()}}},
    {"u_ConstrIndirectArgs", RHI::BindingResource{{indir->buffer.get(), 0, indir->buffer->size()}}},
  });
  
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroupsIndirect(indir->buffer.get(), sizeof(uint32_t) * 8);
  encoder->end();
}

TreeInternalNodes::TreeInternalNodes(VPL::VPLSpawnInfo* info) : vplInfo(info) {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "sst/tree-internal-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto TreeInternalNodes::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("TreeNodes")
      .isBuffer().withSize(sizeof(TreeNode) * (vplInfo->maxNumber * 2 - 1))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("LeafCodes")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("IndirectArgs")
      .isBuffer().withSize(sizeof(TreeConstrIndirectArgs) * 1)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                  (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                              (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT |
                             (uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));
  return reflector;
}

auto TreeInternalNodes::execute(RDG::RenderContext* context,
                             RDG::RenderData const& renderData) noexcept -> void {
  GFX::Buffer* nodes = renderData.getBuffer("TreeNodes");
  GFX::Buffer* codes = renderData.getBuffer("LeafCodes");
  GFX::Buffer* indir = renderData.getBuffer("IndirectArgs");

  updateBindings(context, {
    {"u_Nodes", RHI::BindingResource{{nodes->buffer.get(), 0, nodes->buffer->size()}}},
    {"u_Codes", RHI::BindingResource{{codes->buffer.get(), 0, codes->buffer->size()}}},
    {"u_ConstrIndirectArgs", RHI::BindingResource{{indir->buffer.get(), 0, indir->buffer->size()}}},
  });
  
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->dispatchWorkgroupsIndirect(indir->buffer.get(), sizeof(uint32_t) * 4);
  encoder->end();
}

TreeMergeNodes::TreeMergeNodes(VPL::VPLSpawnInfo* info) : vplInfo(info) {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "sst/tree-merge-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto TreeMergeNodes::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("TreeNodes")
      .isBuffer().withSize(sizeof(TreeNode) * (vplInfo->maxNumber * 2 - 1))
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("VPLMerges")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("VPLData")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("IndirectArgs")
      .isBuffer().withSize(sizeof(TreeConstrIndirectArgs) * 1)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                  (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                              (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT |
                             (uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));
  return reflector;
}

auto TreeMergeNodes::execute(RDG::RenderContext* context,
                             RDG::RenderData const& renderData) noexcept -> void {
  GFX::Buffer* nodes = renderData.getBuffer("TreeNodes");
  GFX::Buffer* merge = renderData.getBuffer("VPLMerges");
  GFX::Buffer* vplda = renderData.getBuffer("VPLData");
  GFX::Buffer* indir = renderData.getBuffer("IndirectArgs");

  updateBindings(context, {
    {"u_Nodes", RHI::BindingResource{{nodes->buffer.get(), 0, nodes->buffer->size()}}},
    {"u_Merge", RHI::BindingResource{{merge->buffer.get(), 0, merge->buffer->size()}}},
    {"u_VPLData", RHI::BindingResource{{vplda->buffer.get(), 0, vplda->buffer->size()}}},
    {"u_ConstrIndirectArgs", RHI::BindingResource{{indir->buffer.get(), 0, indir->buffer->size()}}},
  });
  
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&params, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(ApproxParams));
  encoder->dispatchWorkgroupsIndirect(indir->buffer.get(), sizeof(uint32_t) * 4);
  encoder->end();
}

auto TreeMergeNodes::renderUI() noexcept -> void {
  ImGui::DragFloat("Min Normal Score", &params.minNormalScore, 0.05);
  ImGui::DragFloat("Min Normal Zstd", &params.maxNormalZStd, 0.05);
}

TreeVisualizePass::TreeVisualizePass() {
  auto [vert, frag] = GFX::ShaderLoader_SLANG::load<2u>(
      "../Engine/Shaders/SRenderer/addon/"
      "sst/tree-visualize.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 2>{
          std::make_pair("vertexMain", RHI::ShaderStages::VERTEX),
          std::make_pair("fragmentMain", RHI::ShaderStages::FRAGMENT),
      });

  RenderPass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert),
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(frag));
}

auto TreeVisualizePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("VPLData")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT |
                        (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT));
  reflector.addInput("TreeNodes")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT));
  reflector.addInput("IndirectArgs")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                  (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(RDG::BufferInfo::ConsumeEntry{}
                   .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                              (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
                   .addStage((uint32_t)RHI::PipelineStages::VERTEX_SHADER_BIT |
                             (uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));
  reflector.addOutput("Color")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA8_UNORM)
      .withUsages((uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT)
      .consume(RDG::TextureInfo::ConsumeEntry{
          RDG::TextureInfo::ConsumeType::ColorAttachment}
                   .setAttachmentLoc(0));
  reflector.addInternal("Depth")
      .isTexture().withSize(Math::vec3(1, 1, 1))
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
  GFX::Buffer* vplBuffer = renderData.getBuffer("VPLData");
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

  updateBindings(context, {
    {"CameraBuffer", renderData.getBindingResource("GlobalUniforms").value()},
    {"u_VPLData", RHI::BindingResource{{vplBuffer->buffer.get(), 0, vplBuffer->buffer->size()}}},
    {"u_Nodes", RHI::BindingResource{{tn->buffer.get(), 0, tn->buffer->size()}}},
    {"u_ConstrIndirectArgs", RHI::BindingResource{{ia->buffer.get(), 0, ia->buffer->size()}}},
  });

  pConst.vplIdOffset = 0;

  RHI::RenderPassEncoder* encoder = beginPass(context, color);
  encoder->pushConstants(&pConst,
                         (uint32_t)RHI::ShaderStages::VERTEX |
                             (uint32_t)RHI::ShaderStages::FRAGMENT,
                         0, sizeof(PushConstantBuffer));
  encoder->drawIndirect(ia->buffer.get(), 12 * sizeof(uint32_t), 1, 16);
  encoder->end();
}

auto TreeVisualizePass::renderUI() noexcept -> void {
  ImGui::DragFloat("Color scale", &pConst.gVPLColorScale, 0.01);
  ImGui::DragFloat("Size scale", &pConst.gVPLRenderScale, 0.01);
}

SSTGIPass::SSTGIPass(VPL::VPLSpawnInfo* info) : info(info) {
  auto [rgen] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "sst/sst-vpl-gi.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("RgenMain", RHI::ShaderStages::RAYGEN),
      });

  GFX::SBTsDescriptor sbt = RTCommon::get()->getSBTDescriptor();
  sbt.rgenSBT = GFX::SBTsDescriptor::RayGenerationSBT{
      {Core::ResourceManager::get()->getResource<GFX::ShaderModule>(rgen)}};

  RayTracingPass::init(sbt, 1);
}

auto SSTGIPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("VBuffer")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_UINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Color")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addOutput("Debug")
      .isTexture().withSize(Math::vec3(1, 1, 1))
      .withFormat(RHI::TextureFormat::RGBA32_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("VPLData")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  reflector.addInput("CounterBuffer")
      .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::RAY_TRACING_SHADER_BIT_KHR));
  return reflector;
}

auto SSTGIPass::execute(RDG::RenderContext* context,
                        RDG::RenderData const& renderData) noexcept -> void {
  // Bind common for RT
  std::vector<RHI::BindGroupEntry>* set_0_entries =
      renderData.getBindGroupEntries("CommonScene");
  getBindGroup(context, 0)->updateBinding(*set_0_entries);
  std::vector<RHI::BindGroupEntry>* set_1_entries =
      renderData.getBindGroupEntries("CommonRT");
  getBindGroup(context, 1)->updateBinding(*set_1_entries);

  GFX::Texture* vbuffer = renderData.getTexture("VBuffer");
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* debug = renderData.getTexture("Debug");
  GFX::Buffer* vpl = renderData.getBuffer("VPLData");
  GFX::Buffer* counter = renderData.getBuffer("CounterBuffer");

  updateBindings(context, {
    {"u_vBuffer", RHI::BindingResource{{vbuffer->getUAV(0, 0, 1)}}},
    {"u_color", RHI::BindingResource{{color->getUAV(0, 0, 1)}}},
    {"u_debug", RHI::BindingResource{{debug->getUAV(0, 0, 1)}}},
    {"u_VPLData", RHI::BindingResource{{vpl->buffer.get(), 0, vpl->buffer->size()}}},
    {"u_vplCounter", RHI::BindingResource{{counter->buffer.get(), 0, counter->buffer->size()}}},
  });

  RHI::RayTracingPassEncoder* encoder = beginPass(context);

  struct PushConstant {
    Math::uvec2 resolution;
    uint32_t sample_batch;
    float inv_vpl_paths;
    int spp;
  } pConst = {{vbuffer->texture->width(), vbuffer->texture->height()},
              renderData.getUInt("FrameIdx"),
              1.f / info->maxNumber,
              spp};

  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::RAYGEN, 0,
                         sizeof(PushConstant));
  encoder->traceRays(pConst.resolution.x, pConst.resolution.y, 1);

  encoder->end();
}

auto SSTGIPass::renderUI() noexcept -> void {
    ImGui::DragInt("SPP", &spp);
}

SSTTestGraph::SSTTestGraph() {
  // spawn info
  spawn_info.maxDepth = 2;
  spawn_info.maxNumber = 1024 * 1024;
  // sort setting config
  sort_info.element_count = spawn_info.maxNumber * spawn_info.maxDepth;
  sort_info.type = BitonicSort::BitonicSortSetting::ElementType::UINT64;
  sort_info.dispath = BitonicSort::BitonicSortSetting::DispathType::DYNAMIC_INDIRECT;

  // counter invalid pass
  addPass(std::make_unique<Addon::VPL::CounterInvalidPass>(&spawn_info), "CounterInvalid Pass");
  // spawn pass
  addPass(std::make_unique<Addon::VPL::VPLSpawnPass>(&spawn_info), "VPLSpawn Pass");
  addEdge("CounterInvalid Pass", "CounterBuffer", "VPLSpawn Pass", "CounterBuffer");

  // moton coding pass
  addPass(std::make_unique<TreeEncodePass>(&spawn_info), "TreeEncode Pass");
  addEdge("VPLSpawn Pass", "CounterBuffer", "TreeEncode Pass", "CounterBuffer");
  addEdge("VPLSpawn Pass", "VPLPositions", "TreeEncode Pass", "VPLPositions");
  addEdge("VPLSpawn Pass", "VPLNormals", "TreeEncode Pass", "VPLNormals");
  addEdge("VPLSpawn Pass", "VPLColors", "TreeEncode Pass", "VPLColors");

  // sort pass
  addSubgraph(std::make_unique<Addon::BitonicSort::BitonicSort>(&sort_info), "SortMorton Pass");
  addEdge("TreeEncode Pass", "LeafCodes", "SortMorton Pass", "Input");
  addEdge("VPLSpawn Pass", "CounterBuffer", "SortMorton Pass", "CounterBuffer");

  // tree init pass
  addPass(std::make_unique<TreeInitPass>(&spawn_info), "TreeInit Pass");
  addEdge("TreeEncode Pass", "IndirectArgs", "TreeInit Pass", "IndirectArgs");
  // tree leaf assign pass
  addPass(std::make_unique<TreeAssignLeafIndex>(&spawn_info), "TreeAssignLeaf Pass");
  addEdge("SortMorton Pass", "Output", "TreeAssignLeaf Pass", "SortedCodes");
  addEdge("TreeInit Pass", "TreeNodes", "TreeAssignLeaf Pass", "TreeNodes");
  addEdge("TreeEncode Pass", "IndirectArgs", "TreeAssignLeaf Pass", "IndirectArgs");
  // tree leaf assign pass
  addPass(std::make_unique<TreeInternalNodes>(&spawn_info), "TreeInternal Pass");
  addEdge("SortMorton Pass", "Output", "TreeInternal Pass", "LeafCodes");
  addEdge("TreeAssignLeaf Pass", "TreeNodes", "TreeInternal Pass", "TreeNodes");
  addEdge("TreeEncode Pass", "IndirectArgs", "TreeInternal Pass", "IndirectArgs");
  // tree merge pass
  addPass(std::make_unique<TreeMergeNodes>(&spawn_info), "TreeMerge Pass");
  addEdge("TreeInternal Pass", "TreeNodes", "TreeMerge Pass", "TreeNodes");
  addEdge("TreeInit Pass", "VPLMerges", "TreeMerge Pass", "VPLMerges");
  addEdge("TreeEncode Pass", "VPLData", "TreeMerge Pass", "VPLData");
  addEdge("TreeEncode Pass", "IndirectArgs", "TreeMerge Pass", "IndirectArgs");

  // visualize SST
  bool visualize_sst = false;
  if (visualize_sst) {
    addPass(std::make_unique<TreeVisualizePass>(), "SSTVis Pass");
    addEdge("TreeMerge Pass", "TreeNodes", "SSTVis Pass", "TreeNodes");
    addEdge("TreeMerge Pass", "VPLData", "SSTVis Pass", "VPLData");
    addEdge("TreeEncode Pass", "IndirectArgs", "SSTVis Pass", "IndirectArgs");
  }
  // VBuffer Pass
  addPass(std::make_unique<Addon::VBuffer::RayTraceVBuffer>(), "VBuffer Pass");

  addPass(std::make_unique<SSTGIPass>(&spawn_info), "SSTGI Pass");
  addEdge("TreeMerge Pass", "VPLData", "SSTGI Pass", "VPLData");
  addEdge("VBuffer Pass", "VBuffer", "SSTGI Pass", "VBuffer");
  addEdge("VPLSpawn Pass", "CounterBuffer", "SSTGI Pass", "CounterBuffer");
  
  addPass(std::make_unique<Postprocess::AccumulatePass>(), "Accumulation Pass");
  addEdge("SSTGI Pass", "Color", "Accumulation Pass", "Input");

  addPass(std::make_unique<Addon::Postprocess::ToneMapperPass>(), "ToneMapper Pass");
  addEdge("Accumulation Pass", "Output", "ToneMapper Pass", "Input");

  markOutput("ToneMapper Pass", "Output");
}
}