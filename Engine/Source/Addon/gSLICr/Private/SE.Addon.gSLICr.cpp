#include "../Public/SE.Addon.gSLICr.hpp"
#include <SE.RHI.Profiler.hpp>

namespace SIByL::Addon::gSLICr {
InitClusterCenterPass::InitClusterCenterPass(gSLICrSetting* desc)
    : desc(desc) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/gSLICr/"
      "init-cluster-centers.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto InitClusterCenterPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("Color")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("SPixelInfo")
      .isBuffer()
      .withSize(desc->map_size.x * desc->map_size.y * sizeof(float) * 8)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("IndexImage")
      .isTexture()
      .withSize(Math::ivec3(desc->img_size.x, desc->img_size.y, 1))
      .withFormat(RHI::TextureFormat::R32_SINT);
  return reflector;
}

auto InitClusterCenterPass::execute(RDG::RenderContext* context,
                                    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* sp = renderData.getBuffer("SPixelInfo");
  GFX::Texture* color = renderData.getTexture("Color");

  updateBinding(context, "u_outSPixel", RHI::BindingResource{{sp->buffer.get(), 0, sp->buffer->size()}});
  updateBinding(context, "u_inImg", RHI::BindingResource{color->getSRV(0, 1, 0, 1)});

  struct PushConstant {
    Math::ivec2 map_size;
    Math::ivec2 img_size;
    int spixel_size;
  } pConst;
  pConst.img_size = desc->img_size;
  pConst.map_size = desc->map_size;
  pConst.spixel_size = desc->spixel_size;

  Singleton<RHI::DeviceProfilerManager>::instance()->beginSegment(
      context->cmdEncoder, RHI::PipelineStages::TOP_OF_PIPE_BIT,
      "superpixel-clustering");

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(PushConstant));
  encoder->dispatchWorkgroups((desc->map_size.x + 15) / 16,
                              (desc->map_size.y + 15) / 16, 1);
  encoder->end();
}

auto InitClusterCenterPass::renderUI() noexcept -> void {
  ImGui::DragFloat("CoH weight", &desc->coh_weight, 0.1f, 0, 1);
}

FindCenterAssociationPass::FindCenterAssociationPass(gSLICrSetting* desc)
    : desc(desc) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/gSLICr/"
      "find-center-association.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto FindCenterAssociationPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("Color")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("SPixelInfo")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("IndexImage")
      .isTexture()
      .withFormat(RHI::TextureFormat::R32_SINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("FuzzyWeight")
      .isTexture()
      .withSize(Math::vec3(1))
      .withFormat(RHI::TextureFormat::RGBA16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("FuzzyIndex")
      .isTexture()
      .withSize(Math::vec3(1))
      .withFormat(RHI::TextureFormat::RGBA16_SINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto FindCenterAssociationPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* sp = renderData.getBuffer("SPixelInfo");
  GFX::Texture* ii = renderData.getTexture("IndexImage");
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* fweight = renderData.getTexture("FuzzyWeight");
  GFX::Texture* findex = renderData.getTexture("FuzzyIndex");

  updateBinding(
      context, "u_inSPixel",
      RHI::BindingResource{{sp->buffer.get(), 0, sp->buffer->size()}});
  updateBinding(context, "u_inImg",
                RHI::BindingResource{color->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_outIdxImg",
                RHI::BindingResource{ii->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_fuzzyWeight",
                RHI::BindingResource{fweight->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_fuzzyIDX",
                RHI::BindingResource{findex->getSRV(0, 1, 0, 1)});

  float max_xy_dist = 1.0f / (1.4242f * desc->spixel_size);
  float max_color_dist = 5.0f / (1.7321f);
  struct PushConstant {
    Math::ivec2 map_size;
    Math::ivec2 img_size;
    int spixel_size;
    float weight;
    float max_xy_dist;
    float max_color_dist;
  } pConst;
  pConst.img_size = desc->img_size;
  pConst.map_size = desc->map_size;
  pConst.spixel_size = desc->spixel_size;
  pConst.weight = desc->coh_weight;
  pConst.max_xy_dist = max_xy_dist * max_xy_dist;
  pConst.max_color_dist = max_color_dist * max_color_dist;

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((desc->img_size.x + 15) / 16,
                              (desc->img_size.y + 15) / 16, 1);
  encoder->end();

  Singleton<RHI::DeviceProfilerManager>::instance()->endSegment(
      context->cmdEncoder, RHI::PipelineStages::BOTTOM_OF_PIPE_BIT,
      "superpixel-clustering");
}

//gSLICrGraph::gSLICrGraph(gSLICrSetting const& desc) : desc(desc) {}
//
//auto gSLICrGraph::alias() noexcept -> RDG::AliasDict { 
//  RDG::AliasDict dict;
//  return dict;
//}
//
//auto gSLICrGraph::onRegister(RDG::Graph* graph) noexcept -> void {
//  graph->addPass(std::make_unique<InitClusterCenterPass>(desc),
//                 CONCAT("InitClusterCenter"));
//  // Initialize center association pass
//  graph->addPass(std::make_unique<FindCenterAssociationPass>(desc),
//                 CONCAT("FindCenterAssociation-0"));
//  graph->addEdge(CONCAT("InitClusterCenter"), "SPixelInfo",
//                 CONCAT("FindCenterAssociation-0"), "SPixelInfo");
//  graph->addEdge(CONCAT("InitClusterCenter"), "IndexImage",
//                 CONCAT("FindCenterAssociation-0"), "IndexImage");
//  graph->addEdge(CONCAT("InitClusterCenter"), "Color",
//                 CONCAT("FindCenterAssociation-0"), "Color");
//
//  std::string prev_name = CONCAT("FindCenterAssociation-0");
//  for (int i = 0; i < desc.number_iter; ++i) {
//    std::string ucc_name =
//        CONCAT("UpdateClusterCeter-" + std::to_string(i + 1));
//    std::string frr_name =
//        CONCAT("FinalizeReductionResult-" + std::to_string(i + 1));
//    std::string fca_name =
//        CONCAT("FindCenterAssociationPass-" + std::to_string(i + 1));
//
//    graph->addPass(std::make_unique<UpdateClusterCenterPass>(desc), ucc_name);
//    graph->addEdge(CONCAT("InitClusterCenter"), "Color", ucc_name, "Color");
//    graph->addEdge(prev_name, "IndexImage", ucc_name, "IndexImage");
//    graph->addEdge(CONCAT("InitClusterCenter"), "AccumMap", ucc_name,
//                   "AccumMap");
//
//    graph->addPass(std::make_unique<FinalizeReductionResultPass>(desc),
//                   frr_name);
//    graph->addEdge(ucc_name, "AccumMap", frr_name, "AccumMap");
//    graph->addEdge(ucc_name, "IndexImage", frr_name, "IndexImage");
//
//
//    graph->addPass(std::make_unique<FindCenterAssociationPass>(desc), fca_name);
//    graph->addEdge(frr_name, "AccumMap", fca_name, "AccumMap");
//    graph->addEdge(frr_name, "IndexImage", fca_name, "IndexImage");
//
//
//  }
//  if (desc.enforce_connectivity) {
//    graph->addPass(std::make_unique<EnforceConnectivityPass>(desc),
//                   CONCAT("InitClusterCenter"));
//    graph->addPass(std::make_unique<EnforceConnectivityPass>(desc),
//                   CONCAT("InitClusterCenter"));
//  }
//}

VisualizeSPixelPass::VisualizeSPixelPass(gSLICrSetting const& desc)
    : desc(desc) {
  auto [comp] = GFX::ShaderLoader_SLANG::load(
      "../Engine/Shaders/SRenderer/addon/gSLICr/"
      "draw-superpixel-boundary.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      });

  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto VisualizeSPixelPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("Color")
      .isTexture()
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("IndexImage")
      .isTexture()
      .withFormat(RHI::TextureFormat::R32_SINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("FuzzyWeight")
      .isTexture()
      .withSize(Math::vec3(1))
      .withFormat(RHI::TextureFormat::RGBA16_FLOAT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInput("FuzzyIndex")
      .isTexture()
      .withSize(Math::vec3(1))
      .withFormat(RHI::TextureFormat::RGBA16_SINT)
      .withUsages((uint32_t)RHI::TextureUsage::STORAGE_BINDING)
      .consume(
          RDG::TextureInfo::ConsumeEntry{
              RDG::TextureInfo::ConsumeType::StorageBinding}
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto VisualizeSPixelPass::execute(
    RDG::RenderContext* context, RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Texture* ii = renderData.getTexture("IndexImage");
  GFX::Texture* color = renderData.getTexture("Color");
  GFX::Texture* fweight = renderData.getTexture("FuzzyWeight");
  GFX::Texture* findex = renderData.getTexture("FuzzyIndex");

  updateBinding(context, "u_color",
                RHI::BindingResource{color->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_idxImg",
                RHI::BindingResource{ii->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_fuzzyWeight",
                RHI::BindingResource{fweight->getSRV(0, 1, 0, 1)});
  updateBinding(context, "u_fuzzyIDX",
                RHI::BindingResource{findex->getSRV(0, 1, 0, 1)});

  float max_xy_dist = 1.0f / (1.4242f * desc.spixel_size);
  float max_color_dist = 5.0f / (1.7321f);
  struct PushConstant {
    Math::ivec2 img_size;
    int debug_mode;
  } pConst;
  pConst.img_size = desc.img_size;
  pConst.debug_mode = debug_mode;
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  if (drawBoundary)
    encoder->dispatchWorkgroups((desc.img_size.x + 15) / 16,
                                (desc.img_size.y + 15) / 16, 1);
  encoder->end();
}

auto VisualizeSPixelPass::renderUI() noexcept -> void {
  ImGui::Checkbox("Boundary Draw", &drawBoundary);
  ImGui::DragInt("Debug Mode", &debug_mode, 1, 0, 3);
}

}  // namespace SIByL::Addon::gSLICr