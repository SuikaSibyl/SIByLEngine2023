#include "SE.Addon.RadixForest.hpp"

namespace SIByL::Addon::PrefixSum {
auto FloatDummyPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("InputData")
    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

SinglePassPrefixSumFloatInitPass::SinglePassPrefixSumFloatInitPass(int max_num)
    : max_num(max_num) {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/prefixsum/"
      "singlepass-prefixsum-float-init.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),});
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto SinglePassPrefixSumFloatInitPass::reflect() noexcept -> RDG::PassReflection {
  const int workgroup_count = (max_num + 1023) / 1024;
  RDG::PassReflection reflector;
  reflector.addOutput("PrefixSum")
    .isBuffer() .withSize(sizeof(float) * (max_num + 1))
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("WorkgroupAggregate")
    .isBuffer().withSize(sizeof(float) * workgroup_count)
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("WorkgroupPrefixsum")
    .isBuffer().withSize(sizeof(float) * workgroup_count)
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto SinglePassPrefixSumFloatInitPass::execute(
  RDG::RenderContext* context,
  RDG::RenderData const& renderData) noexcept -> void 
{
  GFX::Buffer* prefixsum = renderData.getBuffer("PrefixSum");
  GFX::Buffer* wg_aggregate = renderData.getBuffer("WorkgroupAggregate");
  GFX::Buffer* wg_prefixsum = renderData.getBuffer("WorkgroupPrefixsum");

  updateBindings(context, {
    {"u_prefix_sum", RHI::BindingResource{{prefixsum->buffer.get(), 0, prefixsum->buffer->size()}}},
    {"u_wg_aggregate", RHI::BindingResource{{wg_aggregate->buffer.get(), 0, wg_aggregate->buffer->size()}}},
    {"u_wg_prefixsum", RHI::BindingResource{{wg_prefixsum->buffer.get(), 0, wg_prefixsum->buffer->size()}}},
  });

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&max_num, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(max_num));
  encoder->dispatchWorkgroups((max_num + 1023) / 1024, 1, 1);
  encoder->end();
}

SinglePassPrefixSumFloatMainPass::SinglePassPrefixSumFloatMainPass(int max_num)
    : max_num(max_num) {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/prefixsum/"
      "singlepass-prefixsum-float-main.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),});
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto SinglePassPrefixSumFloatMainPass::reflect() noexcept -> RDG::PassReflection {
  const int workgroup_count = (max_num + 1023) / 1024;
  RDG::PassReflection reflector;
  reflector.addInput("InputData")
    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("PrefixSum")
    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("WorkgroupAggregate")
    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("WorkgroupPrefixsum")
    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto SinglePassPrefixSumFloatMainPass::execute(
  RDG::RenderContext* context,
  RDG::RenderData const& renderData) noexcept -> void 
{
  GFX::Buffer* inputdata = renderData.getBuffer("InputData");
  GFX::Buffer* prefixsum = renderData.getBuffer("PrefixSum");
  GFX::Buffer* wg_aggregate = renderData.getBuffer("WorkgroupAggregate");
  GFX::Buffer* wg_prefixsum = renderData.getBuffer("WorkgroupPrefixsum");

  updateBindings(context, {
    {"u_data", RHI::BindingResource{{inputdata->buffer.get(), 0, inputdata->buffer->size()}}},
    {"u_prefix_sum", RHI::BindingResource{{prefixsum->buffer.get(), 0, prefixsum->buffer->size()}}},
    {"u_wg_aggregate", RHI::BindingResource{{wg_aggregate->buffer.get(), 0, wg_aggregate->buffer->size()}}},
    {"u_wg_prefixsum", RHI::BindingResource{{wg_prefixsum->buffer.get(), 0, wg_prefixsum->buffer->size()}}},
  });

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&max_num, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(max_num));
  encoder->dispatchWorkgroups((max_num + 1023) / 1024, 1, 1);
  encoder->end();
}
}

namespace SIByL::Addon::RadixForest {
PmfNormalizePass::PmfNormalizePass(int max_num) : max_num(max_num) {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
    "../Engine/Shaders/SRenderer/addon/radixforest/pmf-normalization.slang",
    std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
      std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),});
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto PmfNormalizePass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("PrefixSum")
    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("PMF")
    .isBuffer().withSize(sizeof(float) * max_num)
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto PmfNormalizePass::execute(
  RDG::RenderContext* context,
  RDG::RenderData const& renderData) noexcept -> void 
{
  GFX::Buffer* prefixsum = renderData.getBuffer("PrefixSum");
  GFX::Buffer* pmf = renderData.getBuffer("PMF");

  updateBindings(context, {
    {"u_pmf", RHI::BindingResource{{pmf->buffer.get(), 0, pmf->buffer->size()}}},
    {"u_prefixsum", RHI::BindingResource{{prefixsum->buffer.get(), 0, prefixsum->buffer->size()}}},
  });

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&max_num, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(max_num));
  encoder->dispatchWorkgroups((max_num + 511) / 512, 1, 1);
  encoder->end();
}

RadixForestClearPass::RadixForestClearPass(int max_num, int partition_num)
    : max_num(max_num), partition_num(partition_num) {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
    "../Engine/Shaders/SRenderer/addon/radixforest/radix-forest-clear.slang",
    std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
      std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),});
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto RadixForestClearPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("Otherbounds")
    .isBuffer().withSize(sizeof(int32_t) * (max_num + 1))
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("Nodes")
    .isBuffer().withSize(sizeof(int32_t) * 2 * (max_num + 1))
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto RadixForestClearPass::execute(
  RDG::RenderContext* context,
  RDG::RenderData const& renderData) noexcept -> void 
{
  GFX::Buffer* otherbounds = renderData.getBuffer("Otherbounds");
  GFX::Buffer* nodes = renderData.getBuffer("Nodes");

  updateBindings(context, {
    {"u_otherbounds", RHI::BindingResource{{otherbounds->buffer.get(), 0, otherbounds->buffer->size()}}},
    {"u_nodes", RHI::BindingResource{{nodes->buffer.get(), 0, nodes->buffer->size()}}},
  });

  uint32_t max_nodes = max_num + 1;
  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&max_nodes, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(max_nodes));
  encoder->dispatchWorkgroups((max_nodes + 511) / 512, 1, 1);
  encoder->end();
}

RadixForestConstrPass::RadixForestConstrPass(int max_num, int partition_num)
    : max_num(max_num), partition_num(partition_num) {
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
    "../Engine/Shaders/SRenderer/addon/radixforest/radix-forest-constr.slang",
    std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
      std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),});
  RDG::ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto RadixForestConstrPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInput("PMF")
    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("Otherbounds")
    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
    .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT
              |(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addInputOutput("Nodes")
    .isBuffer().withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
    .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT
              |(uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("HashTable")
    .isBuffer().withSize(sizeof(int32_t) * partition_num)
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("DBG")
    .isBuffer().withSize(sizeof(int32_t) * 4 * (max_num + 1))
    .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
    .consume(RDG::BufferInfo::ConsumeEntry{}
      .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
      .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto RadixForestConstrPass::execute(
  RDG::RenderContext* context,
  RDG::RenderData const& renderData) noexcept -> void 
{
  GFX::Buffer* pmf = renderData.getBuffer("PMF");
  GFX::Buffer* otherbounds = renderData.getBuffer("Otherbounds");
  GFX::Buffer* nodes = renderData.getBuffer("Nodes");
  GFX::Buffer* htable = renderData.getBuffer("HashTable");
  GFX::Buffer* dbg = renderData.getBuffer("DBG");

  updateBindings(context, {
    {"u_pmf", RHI::BindingResource{{pmf->buffer.get(), 0, pmf->buffer->size()}}},
    {"u_otherbounds", RHI::BindingResource{{otherbounds->buffer.get(), 0, otherbounds->buffer->size()}}},
    {"u_nodes", RHI::BindingResource{{nodes->buffer.get(), 0, nodes->buffer->size()}}},
    {"u_hashtable", RHI::BindingResource{{htable->buffer.get(), 0, htable->buffer->size()}}},
    //{"u_debug", RHI::BindingResource{{dbg->buffer.get(), 0, dbg->buffer->size()}}},
  });

  struct PushCosntant {
    int32_t max_num;
    int32_t partition_num;
  } pConst = {
    max_num,
    partition_num
  };

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0, sizeof(pConst));
  encoder->dispatchWorkgroups((max_num + 511) / 512, 1, 1);
  encoder->end();
}

RadixForestBuildGraph::RadixForestBuildGraph(int max_num, int partition_num) {
  addPass(std::make_unique<PrefixSum::FloatDummyPass>(), "FloatInput Pass");
  addPass(std::make_unique<PrefixSum::SinglePassPrefixSumFloatInitPass>(max_num), "PrefixSumInit Pass");
  addPass(std::make_unique<PrefixSum::SinglePassPrefixSumFloatMainPass>(max_num), "PrefixSumMain Pass");
  addEdge("FloatInput Pass", "InputData", "PrefixSumMain Pass", "InputData");
  addEdge("PrefixSumInit Pass", "PrefixSum", "PrefixSumMain Pass", "PrefixSum");
  addEdge("PrefixSumInit Pass", "WorkgroupAggregate", "PrefixSumMain Pass", "WorkgroupAggregate");
  addEdge("PrefixSumInit Pass", "WorkgroupPrefixsum", "PrefixSumMain Pass", "WorkgroupPrefixsum");
  addPass(std::make_unique<RadixForest::PmfNormalizePass>(max_num), "PmfNormalize Pass");
  addEdge("PrefixSumMain Pass", "PrefixSum", "PmfNormalize Pass", "PrefixSum");
  addPass(std::make_unique<RadixForest::RadixForestClearPass>(max_num, partition_num), "RadixForestClear Pass");
  addPass(std::make_unique<RadixForest::RadixForestConstrPass>(max_num, partition_num), "RadixForestConstr Pass");
  addEdge("PmfNormalize Pass", "PMF", "RadixForestConstr Pass", "PMF");
  addEdge("RadixForestClear Pass", "Otherbounds", "RadixForestConstr Pass", "Otherbounds");
  addEdge("RadixForestClear Pass", "Nodes", "RadixForestConstr Pass", "Nodes");
}
}
