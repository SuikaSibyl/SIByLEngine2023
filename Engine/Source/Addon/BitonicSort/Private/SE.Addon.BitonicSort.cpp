#include "../Public/SE.Addon.BitonicSort.hpp"
#include <SE.Math.ShaderCommon.hpp>

namespace SIByL::Addon::BitonicSort {
inline auto SizeOf(BitonicSortSetting::ElementType type) noexcept->size_t {
  switch (type) {
    case SIByL::Addon::BitonicSort::BitonicSortSetting::ElementType::UINT64:
      return sizeof(uint64_t);
    case SIByL::Addon::BitonicSort::BitonicSortSetting::ElementType::UINT32:
      return sizeof(uint32_t);
    default:
      return 0;
  }
};
inline auto ToString(BitonicSortSetting::ElementType type) noexcept -> std::string {
  switch (type) {
    case SIByL::Addon::BitonicSort::BitonicSortSetting::ElementType::UINT64:
      return "1";
    case SIByL::Addon::BitonicSort::BitonicSortSetting::ElementType::UINT32:
      return "0";
    default:
      return "";
  }
};

uint32_t NextPow2(uint32_t Val) {
  uint32_t Mask = (1 << firstbithigh(Val)) - 1;
  return (Val + Mask) & ~Mask;
}

unsigned long upper_power_of_two(unsigned long v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

auto BitonicSortSetting::generateArgs() noexcept -> void {
  element_count_aligned = upper_power_of_two(element_count);
  auto insert_element = [&](uint32_t offset, uint32_t val) {
    if (dispatch_args.size() <= offset) {
      dispatch_args.resize(offset + 1);
    }
    dispatch_args[offset] = val;
  };

  for (int i = 0; i < 22; ++i) {
    uint32_t list_count = element_count;
    uint32_t tid = i;
    uint32_t k = 2048 << tid;

    // We need one more iteration every time the number of thread groups doubles
    if (k > NextPow2((list_count + 2047) & ~2047)) list_count = 0;

    uint32_t PrevDispatches = tid * (tid + 1) / 2;
    uint32_t Offset = PrevDispatches;

    // Generate outer sort dispatch arguments
    for (uint32_t j = k / 2; j > 1024; j /= 2) {
      // All of the groups of size 2j that are full
      uint32_t CompleteGroups = (list_count & ~(2 * j - 1)) / 2048;

      // Remaining items must only be sorted if there are more than j of them
      uint32_t PartialGroups =
          ((uint32_t)max(int(list_count - CompleteGroups * 2048 - j), 0) +
           1023) /
          1024;

      insert_element(Offset, CompleteGroups + PartialGroups);

      Offset += 1;
    }
    // The inner sort always sorts all groups (rounded up to multiples of 2048)
    insert_element(Offset, (list_count + 2047) / 2048);
  }
}

IndirectArgsPass::IndirectArgsPass(BitonicSortSetting* setting)
    : setting(setting) {
  std::string const type_string = ToString(setting->type);
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "bitonicsort/indirect-args.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      },
      {std::make_pair("KEY_TYPE_ENUM", type_string.c_str())});
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));

  alignedMaxNumElements = setting->element_count_aligned;
  maxIterations = Math::log2Int(std::max(2048u, alignedMaxNumElements)) - 10;
}

auto IndirectArgsPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("CounterBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  reflector.addOutput("IndirectArgs")
      .isBuffer()
      .withSize((22 * 23 / 2) * 12)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE |
                  (uint32_t)RHI::BufferUsage::INDIRECT)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto IndirectArgsPass::execute(
    RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* cb = renderData.getBuffer("CounterBuffer");
  GFX::Buffer* ia = renderData.getBuffer("IndirectArgs");

  updateBinding(
      context, "u_IndirectArgsBuffer",
      RHI::BindingResource{{ia->buffer.get(), 0, ia->buffer->size()}});
  updateBinding(
      context, "u_CounterBuffer",
      RHI::BindingResource{{cb->buffer.get(), 0, cb->buffer->size()}});

  const uint32_t MaxIterations =
      Math::log2Int(std::max(2048u, setting->element_count_aligned)) - 10;
  Math::ivec2 pConst;
  pConst.x = MaxIterations;
  pConst.y = setting->counter_offset;

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst,
                         (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(Math::ivec2));
  encoder->dispatchWorkgroups(1, 1, 1);
  encoder->end();
}

PresortPass::PresortPass(BitonicSortSetting* setting, uint32_t offset)
    : setting(setting), offset(offset) {
  std::string const type_string = ToString(setting->type);
  std::vector<std::pair<char const*, char const*>> macros = {
      std::make_pair("KEY_TYPE_ENUM", type_string.c_str())};
  if (setting->dispath == BitonicSortSetting::DispathType::DYNAMIC_INDIRECT) {
    macros.emplace_back(std::make_pair("INDIRECT_DISPATCH", "1"));
  }

  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "bitonicsort/presort-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      },
      macros);
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto PresortPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("KeyBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  if (setting->dispath == BitonicSortSetting::DispathType::DYNAMIC_INDIRECT) {
    reflector.addInput("CounterBuffer")
        .isBuffer()
        .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
        .consume(
            RDG::BufferInfo::ConsumeEntry{}
                .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
                .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
    reflector.addInput("IndirectArgs")
        .isBuffer()
        .withUsages((uint32_t)RHI::BufferUsage::INDIRECT)
        .consume(
            RDG::BufferInfo::ConsumeEntry{}
                .setAccess(
                    (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
                .addStage((uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));
  }
  return reflector;
}

auto PresortPass::execute(RDG::RenderContext* context,
                             RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* kb = renderData.getBuffer("KeyBuffer");
  
  updateBinding(
      context, "g_SortBuffer",
      RHI::BindingResource{{kb->buffer.get(), 0, kb->buffer->size()}});

  if (setting->dispath == BitonicSortSetting::DispathType::DYNAMIC_INDIRECT) {
    GFX::Buffer* cb = renderData.getBuffer("CounterBuffer");
    GFX::Buffer* ia = renderData.getBuffer("IndirectArgs");
    updateBinding(
        context, "u_CounterBuffer",
        RHI::BindingResource{{cb->buffer.get(), 0, cb->buffer->size()}});

    RHI::ComputePassEncoder* encoder = beginPass(context);
    encoder->pushConstants(&setting->counter_offset,
                           (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                           sizeof(uint32_t));
    encoder->dispatchWorkgroupsIndirect(ia->buffer.get(), offset);
    encoder->end();
  } else {
    RHI::ComputePassEncoder* encoder = beginPass(context);
    encoder->pushConstants(&setting->element_count,
                           (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                           sizeof(uint32_t));
    encoder->dispatchWorkgroups((setting->element_count + 2047) / 2048, 1, 1);
    encoder->end();
  }
}

OuterSortPass::OuterSortPass(BitonicSortSetting* setting, uint32_t k,
                             uint32_t j, uint32_t arg)
    : setting(setting), k(k), j(j), arg(arg) {
  std::string const type_string = ToString(setting->type);
  std::vector<std::pair<char const*, char const*>> macros = {
      std::make_pair("KEY_TYPE_ENUM", type_string.c_str())};
  if (setting->dispath == BitonicSortSetting::DispathType::DYNAMIC_INDIRECT) {
    macros.emplace_back(std::make_pair("INDIRECT_DISPATCH", "1"));
  }

  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "bitonicsort/outersort-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      },
      macros);
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto OuterSortPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("KeyBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  if (setting->dispath == BitonicSortSetting::DispathType::DYNAMIC_INDIRECT) {
    reflector.addInput("CounterBuffer")
        .isBuffer()
        .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
        .consume(
            RDG::BufferInfo::ConsumeEntry{}
                .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
                .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
    reflector.addInput("IndirectArgs")
        .isBuffer()
        .withUsages((uint32_t)RHI::BufferUsage::INDIRECT)
        .consume(
            RDG::BufferInfo::ConsumeEntry{}
                .setAccess(
                    (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
                .addStage((uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));
  }
  return reflector;
}

auto OuterSortPass::execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* kb = renderData.getBuffer("KeyBuffer");
  updateBinding(
      context, "g_SortBuffer",
      RHI::BindingResource{{kb->buffer.get(), 0, kb->buffer->size()}});

  if (setting->dispath == BitonicSortSetting::DispathType::DYNAMIC_INDIRECT) {
    GFX::Buffer* cb = renderData.getBuffer("CounterBuffer");
    GFX::Buffer* ia = renderData.getBuffer("IndirectArgs");
    updateBinding(
        context, "u_CounterBuffer",
        RHI::BindingResource{{cb->buffer.get(), 0, cb->buffer->size()}});

    struct PushConstant {
      uint k;  // k >= 4096
      uint j;  // j >= 2048 && j < k
      uint counterOffset;
    } pConst;
    pConst.k = k;
    pConst.j = j;
    pConst.counterOffset = setting->counter_offset;

    RHI::ComputePassEncoder* encoder = beginPass(context);
    encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                           sizeof(PushConstant));
    encoder->dispatchWorkgroupsIndirect(ia->buffer.get(), arg);
    encoder->end();
  } else {
    struct PushConstant {
      uint k;  // k >= 4096
      uint j;  // j >= 2048 && j < k
      uint numElements;
    } pConst;
    pConst.k = k;
    pConst.j = j;
    pConst.numElements = setting->element_count;

    RHI::ComputePassEncoder* encoder = beginPass(context);
    encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                           sizeof(PushConstant));
    encoder->dispatchWorkgroups(arg, 1, 1);
    encoder->end();
  }
}

InnerSortPass::InnerSortPass(BitonicSortSetting* setting, uint32_t k,
                             uint32_t arg)
    : setting(setting), k(k), arg(arg) {
  std::string const type_string = ToString(setting->type);
  std::vector<std::pair<char const*, char const*>> macros = {
      std::make_pair("KEY_TYPE_ENUM", type_string.c_str())};
  if (setting->dispath == BitonicSortSetting::DispathType::DYNAMIC_INDIRECT) {
    macros.emplace_back(std::make_pair("INDIRECT_DISPATCH", "1"));
  }

  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "bitonicsort/innersort-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      },
      macros);
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto InnerSortPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addInputOutput("KeyBuffer")
      .isBuffer()
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  if (setting->dispath == BitonicSortSetting::DispathType::DYNAMIC_INDIRECT) {
    reflector.addInput("CounterBuffer")
        .isBuffer()
        .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
        .consume(
            RDG::BufferInfo::ConsumeEntry{}
                .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT)
                .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
    reflector.addInput("IndirectArgs")
        .isBuffer()
        .withUsages((uint32_t)RHI::BufferUsage::INDIRECT)
        .consume(
            RDG::BufferInfo::ConsumeEntry{}
                .setAccess(
                    (uint32_t)RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT)
                .addStage((uint32_t)RHI::PipelineStages::DRAW_INDIRECT_BIT));
  }
  return reflector;
}

auto InnerSortPass::execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* kb = renderData.getBuffer("KeyBuffer");

  updateBinding(
      context, "g_SortBuffer",
      RHI::BindingResource{{kb->buffer.get(), 0, kb->buffer->size()}});

  if (setting->dispath == BitonicSortSetting::DispathType::DYNAMIC_INDIRECT) {
    GFX::Buffer* cb = renderData.getBuffer("CounterBuffer");
    GFX::Buffer* ia = renderData.getBuffer("IndirectArgs");
    updateBinding(
        context, "u_CounterBuffer",
        RHI::BindingResource{{cb->buffer.get(), 0, cb->buffer->size()}});

    RHI::ComputePassEncoder* encoder = beginPass(context);
    encoder->pushConstants(&setting->counter_offset,
                           (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                           sizeof(int32_t));
    encoder->dispatchWorkgroupsIndirect(ia->buffer.get(), arg);
    encoder->end();
  } else {
    RHI::ComputePassEncoder* encoder = beginPass(context);
    encoder->pushConstants(&setting->element_count,
                           (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                           sizeof(int32_t));
    encoder->dispatchWorkgroups(arg, 1, 1);
    encoder->end();
  }
}

BitonicSort::BitonicSort(BitonicSortSetting* setting) : setting(setting) {
  if (setting->dispath == BitonicSortSetting::DispathType::DYNAMIC_INDIRECT) {
    this->setting->element_count_aligned =
        upper_power_of_two(setting->element_count);
  } else {
    // STATIC_DIRECT draw dispatch
    setting->generateArgs();    
  }
}

auto BitonicSort::alias() noexcept -> RDG::AliasDict {
  RDG::AliasDict dict;
  dict.addAlias("Input", CONCAT("Presort"), "KeyBuffer");
  dict.addAlias("Output", output_pass, "KeyBuffer");
  if (setting->dispath == BitonicSortSetting::DispathType::DYNAMIC_INDIRECT) {
    dict.addAlias("CounterBuffer", CONCAT("IndirectArgs"), "CounterBuffer");
  }
  return dict;
}

auto BitonicSort::onRegister(RDG::Graph* graph) noexcept -> void {
  uint32_t outer_pass_id = 0;
  uint32_t inner_pass_id = 0;
  uint32_t alignedMaxNumElements = setting->element_count_aligned;
  uint32_t argOffset = 1;

  if (setting->dispath == BitonicSortSetting::DispathType::DYNAMIC_INDIRECT) {
    graph->addPass(std::make_unique<IndirectArgsPass>(setting),
                   CONCAT("IndirectArgs"));
    std::string currentPass;
    std::string previousPass = CONCAT("Presort");
    graph->addPass(std::make_unique<PresortPass>(setting, 0), CONCAT("Presort"));
    graph->addEdge(CONCAT("IndirectArgs"), "CounterBuffer", CONCAT("Presort"),
                   "CounterBuffer");
    graph->addEdge(CONCAT("IndirectArgs"), "IndirectArgs", CONCAT("Presort"),
                   "IndirectArgs");

    uint32_t indirectArgsOffset = 12;

    for (uint32_t k = 4096; k <= alignedMaxNumElements; k *= 2) {
      for (uint32_t j = k / 2; j >= 2048; j /= 2) {
        currentPass = CONCAT("Outer-" + std::to_string(outer_pass_id++));
        graph->addPass(std::make_unique<OuterSortPass>(setting, k, j, indirectArgsOffset),
                       currentPass);
        graph->addEdge(previousPass, "KeyBuffer", currentPass, "KeyBuffer");
        graph->addEdge(CONCAT("IndirectArgs"), "CounterBuffer", currentPass,
                       "CounterBuffer");
        graph->addEdge(CONCAT("IndirectArgs"), "IndirectArgs", currentPass,
                       "IndirectArgs");

        previousPass = currentPass;
        indirectArgsOffset += 12;
      }

      currentPass = CONCAT("Inner-" + std::to_string(inner_pass_id++));
      graph->addPass(std::make_unique<InnerSortPass>(setting, k, indirectArgsOffset),
                     currentPass);
      graph->addEdge(previousPass, "KeyBuffer", currentPass, "KeyBuffer");
      graph->addEdge(CONCAT("IndirectArgs"), "CounterBuffer", currentPass,
                     "CounterBuffer");
      graph->addEdge(CONCAT("IndirectArgs"), "IndirectArgs", currentPass,
                     "IndirectArgs");
      previousPass = currentPass;
      indirectArgsOffset += 12;
    }
    
    output_pass = previousPass;
  } else {
    std::string currentPass;
    std::string previousPass = CONCAT("Presort");
    graph->addPass(std::make_unique<PresortPass>(setting), CONCAT("Presort"));

    for (uint32_t k = 4096; k <= alignedMaxNumElements; k *= 2) {
      for (uint32_t j = k / 2; j >= 2048; j /= 2) {
        currentPass = CONCAT("Outer-" + std::to_string(outer_pass_id++));
        graph->addPass(std::make_unique<OuterSortPass>(
                           setting, k, j, setting->dispatch_args[argOffset]),
                       currentPass);
        graph->addEdge(previousPass, "KeyBuffer", currentPass, "KeyBuffer");
        previousPass = currentPass;
        argOffset += 1;
      }

      currentPass = CONCAT("Inner-" + std::to_string(inner_pass_id++));
      graph->addPass(std::make_unique<InnerSortPass>(
                         setting, k, setting->dispatch_args[argOffset]),
                     currentPass);
      graph->addEdge(previousPass, "KeyBuffer", currentPass, "KeyBuffer");
      previousPass = currentPass;
      argOffset += 1;
    }

    output_pass = previousPass;
  }
}

RandomInitPass::RandomInitPass(BitonicSortSetting* setting) : setting(setting) {
  std::string const type_string = ToString(setting->type);
  auto [comp] = GFX::ShaderLoader_SLANG::load<1u>(
      "../Engine/Shaders/SRenderer/addon/"
      "bitonicsort/random-init-pass.slang",
      std::array<std::pair<std::string, RHI::ShaderStages>, 1>{
          std::make_pair("ComputeMain", RHI::ShaderStages::COMPUTE),
      },
      {std::make_pair("KEY_TYPE_ENUM", type_string.c_str())});
  ComputePass::init(
      Core::ResourceManager::get()->getResource<GFX::ShaderModule>(comp));
}

auto RandomInitPass::reflect() noexcept -> RDG::PassReflection {
  RDG::PassReflection reflector;
  reflector.addOutput("KeyBuffer")
      .isBuffer()
      .withSize(SizeOf(setting->type) * setting->element_count)
      .withUsages((uint32_t)RHI::BufferUsage::STORAGE)
      .consume(
          RDG::BufferInfo::ConsumeEntry{}
              .setAccess((uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT |
                         (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT)
              .addStage((uint32_t)RHI::PipelineStages::COMPUTE_SHADER_BIT));
  return reflector;
}

auto RandomInitPass::execute(RDG::RenderContext* context,
    RDG::RenderData const& renderData) noexcept
    -> void {
  GFX::Buffer* kb = renderData.getBuffer("KeyBuffer");

  updateBinding(
      context, "g_SortBuffer",
      RHI::BindingResource{{kb->buffer.get(), 0, kb->buffer->size()}});

  struct PushConstant {
    uint32_t numElements;
    uint32_t randomSeed;
  } pConst;
  pConst.numElements = setting->element_count;
  pConst.randomSeed = renderData.getUInt("AccumIdx");

  RHI::ComputePassEncoder* encoder = beginPass(context);
  encoder->pushConstants(&pConst, (uint32_t)RHI::ShaderStages::COMPUTE, 0,
                         sizeof(PushConstant));
  encoder->dispatchWorkgroups((setting->element_count + 2047) / 2048, 1, 1);
  encoder->end();
}
}