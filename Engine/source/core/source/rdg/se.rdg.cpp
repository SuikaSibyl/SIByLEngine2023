#define DLIB_EXPORT
#include <se.rdg.hpp>
#undef DLIB_EXPORT

namespace se::rdg {
auto BufferInfo::ConsumeEntry::addStage(rhi::PipelineStages stage) noexcept -> ConsumeEntry& {
  stages |= stage;
  return *this;
}

auto BufferInfo::ConsumeEntry::setAccess(uint32_t acc) noexcept -> ConsumeEntry& {
  access = acc;
  return *this;
}

auto BufferInfo::ConsumeEntry::setSubresource(uint64_t offset, uint64_t size
) noexcept -> BufferInfo::ConsumeEntry& {
  this->offset = offset;
  this->size = size;
  return *this;
}

auto BufferInfo::withSize(uint32_t _size) noexcept -> BufferInfo& {
  size = _size;
  return *this;
}

auto BufferInfo::withUsages(rhi::BufferUsages _usages) noexcept -> BufferInfo& {
  usages = _usages;
  return *this;
}

auto BufferInfo::withFlags(ResourceFlags _flags) noexcept -> BufferInfo& {
  flags = _flags;
  return *this;
}

auto BufferInfo::consume(ConsumeEntry const& entry) noexcept -> BufferInfo& {
  consumeHistories.emplace_back(entry);
  return *this;
}

auto TextureInfo::ConsumeEntry::addStage(rhi::PipelineStages stage) noexcept -> ConsumeEntry& {
    stages |= stage;
    return *this; }

auto TextureInfo::ConsumeEntry::setLayout(rhi::TextureLayout _layout) noexcept -> ConsumeEntry& {
    layout = _layout;
    return *this; }

auto TextureInfo::ConsumeEntry::enableDepthWrite(bool set) noexcept -> ConsumeEntry& {
    depthWrite = set;
    return *this; }

auto TextureInfo::ConsumeEntry::setDepthCompareFn(rhi::CompareFunction fn) noexcept
-> ConsumeEntry& {
    depthCmp = fn;
    return *this; }

auto TextureInfo::ConsumeEntry::setSubresource(uint32_t mip_beg, uint32_t mip_end,
  uint32_t level_beg, uint32_t level_end) noexcept
  -> ConsumeEntry& {
  this->mip_beg = mip_beg;
  this->mip_end = mip_end;
  this->level_beg = level_beg;
  this->level_end = level_end;
  return *this;
}

auto TextureInfo::ConsumeEntry::setAttachmentLoc(uint32_t loc) noexcept -> ConsumeEntry& {
  attachLoc = loc;
  return *this;
}

auto TextureInfo::ConsumeEntry::setAccess(uint32_t acc) noexcept -> ConsumeEntry& {
  access = acc;
  return *this;
}

auto TextureInfo::consume(ConsumeEntry const& _entry) noexcept -> TextureInfo& {
  ConsumeEntry entry = _entry;
  if (entry.type == ConsumeType::ColorAttachment) {
    entry.access |= uint32_t(rhi::AccessFlagBits::COLOR_ATTACHMENT_READ_BIT) |
                    uint32_t(rhi::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT);
    entry.stages |= uint32_t(rhi::PipelineStageBit::COLOR_ATTACHMENT_OUTPUT_BIT);
    entry.layout = rhi::TextureLayout::COLOR_ATTACHMENT_OPTIMAL;
  } else if (entry.type == ConsumeType::DepthStencilAttachment) {
    entry.access |=
        uint32_t(rhi::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT) |
        uint32_t(rhi::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    entry.stages |= uint32_t(rhi::PipelineStageBit::COLOR_ATTACHMENT_OUTPUT_BIT) |
                    uint32_t(rhi::PipelineStageBit::EARLY_FRAGMENT_TESTS_BIT) |
                    uint32_t(rhi::PipelineStageBit::LATE_FRAGMENT_TESTS_BIT);
    entry.layout = rhi::TextureLayout::DEPTH_ATTACHMENT_OPTIMAL;
  } else if (entry.type == ConsumeType::TextureBinding) {
    entry.access |= uint32_t(rhi::AccessFlagBits::SHADER_READ_BIT);
    entry.layout = rhi::TextureLayout::SHADER_READ_ONLY_OPTIMAL;
  } else if (entry.type == ConsumeType::StorageBinding) {
    entry.access |= uint32_t(rhi::AccessFlagBits::SHADER_READ_BIT) |
                    uint32_t(rhi::AccessFlagBits::SHADER_WRITE_BIT);
    entry.layout = rhi::TextureLayout::GENERAL;
  }
  consumeHistories.emplace_back(entry);
  return *this;
}

auto TextureInfo::setInfo(TextureInfo const& x) noexcept
    -> TextureInfo& {
  sizeDef = x.sizeDef;
  if (sizeDef == SizeDefine::Absolute)
    size.absolute = x.size.absolute;
  else if (sizeDef == SizeDefine::Relative)
    size.relative = x.size.relative;
  format = x.format;
  levels = x.levels;
  return *this;
}

auto TextureInfo::withSize(se::ivec3 absolute) noexcept
    -> TextureInfo& {
  sizeDef = SizeDefine::Absolute;
  size.absolute = absolute;
  return *this;
}

auto TextureInfo::withSize(se::vec3 relative) noexcept -> TextureInfo& {
  sizeDef = SizeDefine::Relative;
  size.relative = relative;
  return *this;
}

auto TextureInfo::withSizeRelative(std::string const& src, se::vec3 relative) noexcept -> TextureInfo& {
  sizeDef = SizeDefine::RelativeToAnotherTex;
  sizeRefName = src;
  size.relative = relative;
  return *this;
}

auto TextureInfo::withLevels(uint32_t _levels) noexcept -> TextureInfo& {
  levels = _levels;
  return *this;
}

auto TextureInfo::withLayers(uint32_t _layers) noexcept -> TextureInfo& {
  layers = _layers;
  return *this;
}

auto TextureInfo::withSamples(uint32_t _samples) noexcept
    -> TextureInfo& {
  samples = _samples;
  return *this;
}

auto TextureInfo::withFormat(rhi::TextureFormat _format) noexcept -> TextureInfo& {
  format = _format;
  return *this;
}

auto TextureInfo::withStages(rhi::ShaderStages flags) noexcept -> TextureInfo& {
  sflags = flags;
  return *this;
}

auto TextureInfo::withUsages(rhi::TextureUsages _usages) noexcept -> TextureInfo& {
  usages = _usages;
  if (usages | uint32_t(rhi::TextureUsageBit::COLOR_ATTACHMENT)) {
    stages |= uint32_t(rhi::PipelineStageBit::COLOR_ATTACHMENT_OUTPUT_BIT) | sflags;
    access |= uint32_t(rhi::AccessFlagBits::COLOR_ATTACHMENT_READ_BIT) |
              uint32_t(rhi::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT);
    laytout = rhi::TextureLayout::COLOR_ATTACHMENT_OPTIMAL;
  } else if (usages | uint32_t(rhi::TextureUsageBit::DEPTH_ATTACHMENT)) {
    stages |= uint32_t(rhi::PipelineStageBit::COLOR_ATTACHMENT_OUTPUT_BIT) |
              uint32_t(rhi::PipelineStageBit::EARLY_FRAGMENT_TESTS_BIT) |
              uint32_t(rhi::PipelineStageBit::EARLY_FRAGMENT_TESTS_BIT) | sflags;
    access |= uint32_t(rhi::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT) |
              uint32_t(rhi::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    laytout = rhi::TextureLayout::DEPTH_ATTACHMENT_OPTIMAL;
  } else if (usages | uint32_t(rhi::TextureUsageBit::TEXTURE_BINDING)) {
    stages |= sflags;
    access |= uint32_t(rhi::AccessFlagBits::SHADER_READ_BIT);
    laytout = rhi::TextureLayout::SHADER_READ_ONLY_OPTIMAL;
  } else if (usages | uint32_t(rhi::TextureUsageBit::STORAGE_BINDING)) {
    stages |= sflags;
    access |= uint32_t(rhi::AccessFlagBits::SHADER_READ_BIT) |
              uint32_t(rhi::AccessFlagBits::SHADER_WRITE_BIT);
    laytout = rhi::TextureLayout::GENERAL;
  }
  return *this;
}

auto TextureInfo::withFlags(ResourceFlags _flags) noexcept -> TextureInfo& {
  flags = _flags;
  return *this;
}

auto TextureInfo::getSize(se::ivec3 ref) const noexcept -> rhi::Extend3D {
  if (sizeDef == SizeDefine::Absolute)
    return rhi::Extend3D{uint32_t(size.absolute.x), uint32_t(size.absolute.y),
                         uint32_t(size.absolute.z)};
  else
    return rhi::Extend3D{uint32_t(std::ceil(size.relative.x * ref.x)),
                         uint32_t(std::ceil(size.relative.y * ref.y)),
                         uint32_t(std::ceil(size.relative.z * ref.z))};
}

auto ResourceInfo::isBuffer() noexcept -> BufferInfo& {
  type = Type::Buffer;
  info.buffer = BufferInfo{};
  return info.buffer;
}

auto ResourceInfo::isTexture() noexcept -> TextureInfo& {
  type = Type::Texture;
  info.texture = TextureInfo{};
  return info.texture;
}

PassReflection::PassReflection() {
  inputResources.clear();
  outputResources.clear();
  inputOutputResources.clear();
  internalResources.clear();
}

auto PassReflection::addInput(std::string const& name) noexcept
    -> ResourceInfo& {
  return inputResources[name];
}
auto PassReflection::addOutput(std::string const& name) noexcept
    -> ResourceInfo& {
  return outputResources[name];
}
auto PassReflection::addInputOutput(std::string const& name) noexcept
    -> ResourceInfo& {
  return inputOutputResources[name];
}
auto PassReflection::addInternal(std::string const& name) noexcept
    -> ResourceInfo& {
  return internalResources[name];
}
auto PassReflection::getDepthStencilState() noexcept -> rhi::DepthStencilState {
  std::vector<std::unordered_map<std::string, ResourceInfo>*> pools =
      std::vector<std::unordered_map<std::string, ResourceInfo>*>{
          &inputResources, &outputResources, &inputOutputResources,
          &internalResources};
  for (auto& pool : pools) {
    for (auto& pair : *pool) {
      if (pair.second.type != ResourceInfo::Type::Texture) continue;
      if (rhi::hasDepthBit(pair.second.info.texture.format)) {
        TextureInfo::ConsumeEntry entry;
        for (auto const& ch : pair.second.info.texture.consumeHistories) {
          if (ch.type == TextureInfo::ConsumeType::DepthStencilAttachment) {
            entry = ch;
            break;
          }
        }
        return rhi::DepthStencilState{pair.second.info.texture.format,
                                      entry.depthWrite, entry.depthCmp};
      }
    }
  }
  return rhi::DepthStencilState{};
}
auto PassReflection::getColorTargetState() noexcept
    -> std::vector<rhi::ColorTargetState> {
  std::vector<rhi::ColorTargetState> cts;
  std::vector<std::unordered_map<std::string, ResourceInfo>*> pools =
      std::vector<std::unordered_map<std::string, ResourceInfo>*>{
          &inputResources, &outputResources, &inputOutputResources,
          &internalResources};
  for (auto& pool : pools) {
    for (auto& pair : *pool) {
      if (pair.second.type != ResourceInfo::Type::Texture) continue;
      if (!rhi::hasDepthBit(pair.second.info.texture.format)) {
        for (auto const& history : pair.second.info.texture.consumeHistories) {
          if (history.type != TextureInfo::ConsumeType::ColorAttachment)
            continue;
          cts.resize(history.attachLoc + 1);
          cts[history.attachLoc] = rhi::ColorTargetState{pair.second.info.texture.format};
        }
      }
    }
  }
  return cts;
}

auto PassReflection::getResourceInfo(std::string const& name) noexcept -> ResourceInfo* {
  if (inputResources.find(name) != inputResources.end()) {
    return &inputResources[name];
  } else if (outputResources.find(name) != outputResources.end()) {
    return &outputResources[name];
  } else if (inputOutputResources.find(name) != inputOutputResources.end()) {
    return &inputOutputResources[name];
  } else if (internalResources.find(name) != internalResources.end()) {
    return &internalResources[name];
  }
  root::print::error(std::format(
    "RDG::PassReflection::getResourceInfo Failed to find resource \"{0}\"", name));
  return nullptr;
}

auto toBufferDescriptor(BufferInfo const& info) noexcept -> rhi::BufferDescriptor {
  return rhi::BufferDescriptor{ info.size, info.usages };
}

auto toTextureDescriptor(TextureInfo const& info, se::ivec3 ref_size) noexcept -> rhi::TextureDescriptor {
  rhi::Extend3D size = info.getSize(ref_size);
  return rhi::TextureDescriptor{
    size,
    info.levels,
    info.layers,
    info.samples,
    size.depthOrArrayLayers == 1 ? rhi::TextureDimension::TEX2D
                                 : rhi::TextureDimension::TEX3D,
    info.format,
    info.usages | (uint32_t)rhi::TextureUsageBit::TEXTURE_BINDING |
        (uint32_t)rhi::TextureUsageBit::COPY_SRC,
    std::vector<rhi::TextureFormat>{info.format},
    info.tflags};
}

auto RenderData::setDelegate(
    std::string const& name,
    std::function<void(DelegateData const&)> const& fn) noexcept -> void {
    delegates[name] = fn;
}

auto RenderData::getDelegate(std::string const& name) const noexcept
-> std::function<void(DelegateData const&)> const& {
    auto const& iter = delegates.find(name);
    if (iter == delegates.end()) return nullptr;
    return iter->second;
}

auto RenderData::setBindingResource(
  std::string const& name,
  rhi::BindingResource const& bindGroupEntry) noexcept
  -> void {
  bindingResources[name] = bindGroupEntry;
}

auto RenderData::getBindingResource(std::string const& name) const noexcept
-> std::optional<rhi::BindingResource> {
    auto const& iter = bindingResources.find(name);
    if (iter == bindingResources.end()) return std::nullopt;
    return iter->second;
}

auto RenderData::setBindGroupEntries(
    std::string const& name,
    std::vector<rhi::BindGroupEntry>* bindGroup) noexcept -> void {
    bindGroups[name] = bindGroup;
}

auto RenderData::getBindGroupEntries(std::string const& name) const noexcept
-> std::vector<rhi::BindGroupEntry>* {
    auto const& iter = bindGroups.find(name);
    if (iter == bindGroups.end()) return nullptr;
    return iter->second;
}
//
//auto RenderData::getTexture(std::string const& name) const noexcept -> gfx::Texture* {
//  if (pass->pReflection.inputResources.find(name) !=
//      pass->pReflection.inputResources.end()) {
//    return graph
//        ->textureResources[pass->pReflection.inputResources[name]
//                               .devirtualizeID]
//        ->texture;
//  }
//  if (pass->pReflection.outputResources.find(name) !=
//      pass->pReflection.outputResources.end()) {
//    return graph
//        ->textureResources[pass->pReflection.outputResources[name]
//                               .devirtualizeID]
//        ->texture;
//  }
//  if (pass->pReflection.inputOutputResources.find(name) !=
//      pass->pReflection.inputOutputResources.end()) {
//    return graph
//        ->textureResources[pass->pReflection.inputOutputResources[name]
//                               .devirtualizeID]
//        ->texture;
//  }
//  if (pass->pReflection.internalResources.find(name) !=
//      pass->pReflection.internalResources.end()) {
//    return graph
//        ->textureResources[pass->pReflection.internalResources[name]
//                               .devirtualizeID]
//        ->texture;
//  }
//}
//
//auto RenderData::getBuffer(std::string const& name) const noexcept -> gfx::Buffer* {
//  if (pass->pReflection.inputResources.find(name) !=
//      pass->pReflection.inputResources.end()) {
//    return graph
//        ->bufferResources[pass->pReflection.inputResources[name].devirtualizeID]
//        ->buffer;
//  }
//  if (pass->pReflection.outputResources.find(name) !=
//      pass->pReflection.outputResources.end()) {
//    return graph
//        ->bufferResources[pass->pReflection.outputResources[name]
//                              .devirtualizeID]
//        ->buffer;
//  }
//  if (pass->pReflection.inputOutputResources.find(name) !=
//      pass->pReflection.inputOutputResources.end()) {
//    return graph
//        ->bufferResources[pass->pReflection.inputOutputResources[name]
//                              .devirtualizeID]
//        ->buffer;
//  }
//  if (pass->pReflection.internalResources.find(name) !=
//      pass->pReflection.internalResources.end()) {
//    return graph
//        ->bufferResources[pass->pReflection.internalResources[name]
//                              .devirtualizeID]
//        ->buffer;
//  }
//}

auto RenderData::setUVec2(std::string const& name, se::uvec2 v) noexcept -> void {
  uvec2s[name] = v;
}

auto RenderData::getUVec2(std::string const& name) const noexcept -> se::uvec2 {
  auto const& iter = uvec2s.find(name);
  if (iter == uvec2s.end()) return se::uvec2{ 0 };
  return iter->second;
}

auto RenderData::setUInt(std::string const& name, uint32_t v) noexcept -> void {
  uints[name] = v;
}

auto RenderData::getUInt(std::string const& name) const noexcept -> uint32_t {
  auto const& iter = uints.find(name);
  if (iter == uints.end()) return 0;
  return iter->second;
}

auto RenderData::setPtr(std::string const& name, void* v) noexcept -> void {
  ptrs[name] = v;
}

auto RenderData::getPtr(std::string const& name) const noexcept -> void* {
    auto const& iter = ptrs.find(name);
    if (iter == ptrs.end()) return 0;
    return iter->second;
}

auto RenderData::setMat4(std::string const& name, se::mat4 m) noexcept -> void {
  mat4s[name] = m;
}

auto RenderData::getMat4(std::string const& name) const noexcept -> se::mat4 {
  auto const& iter = mat4s.find(name);
  if (iter == mat4s.end()) return se::mat4();
  return iter->second;
}

auto Pass::generateMarker() noexcept -> void {
  marker.name = identifier;
  marker.color = { 0.490, 0.607, 0.003, 1. };
}

auto Pass::init() noexcept -> void { pReflection = this->reflect(); }

auto AliasDict::addAlias(std::string const& subgraph_resource,
    std::string const& pass,
    std::string const& pass_resource) noexcept
    -> AliasDict& {
    dict[subgraph_resource] = Value{ pass, pass_resource };
    return *this;
}

auto Subgraph::CONCAT(std::string const& name) noexcept -> std::string {
  return identifier + "." + name;
}

auto Subgraph::generateMarker() noexcept -> void {
  marker.name = identifier;
  marker.color = { 0.235, 0.321, 0.2, 1. };
}

auto PipelinePass::getBindGroup(RenderContext* context, uint32_t size) noexcept -> rhi::BindGroup* {
  return bindgroups[size][context->flightIdx].get();
}

auto PipelinePass::init(std::vector<gfx::ShaderModule*> shaderModules) noexcept  -> void {
  Pass::init();
  rhi::Device* device = gfx::GFXContext::device;
  // create pipeline reflection
  for (auto& sm : shaderModules) reflection = reflection + sm->reflection;
  // create bindgroup layouts
  bindgroupLayouts.resize(reflection.bindings.size());
  for (int i = 0; i < reflection.bindings.size(); ++i) {
    rhi::BindGroupLayoutDescriptor desc =
        gfx::ShaderReflection::toBindGroupLayoutDescriptor(reflection.bindings[i]);
    bindgroupLayouts[i] = device->createBindGroupLayout(desc);
  }
  // create bindgroups
  bindgroups.resize(reflection.bindings.size());
  for (size_t i = 0; i < reflection.bindings.size(); ++i) {
    for (size_t j = 0; j < MULTIFRAME_FLIGHTS_COUNT; ++j) {
      bindgroups[i][j] = device->createBindGroup(rhi::BindGroupDescriptor{
          bindgroupLayouts[i].get(),
          std::vector<rhi::BindGroupEntry>{},
      });
    }
  }
  // create pipelineLayout
  rhi::PipelineLayoutDescriptor desc = {};
  desc.pushConstants.resize(reflection.pushConstant.size());
  for (int i = 0; i < reflection.pushConstant.size(); ++i)
    desc.pushConstants[i] = rhi::PushConstantEntry{
      reflection.pushConstant[i].stages,
      reflection.pushConstant[i].offset,
      reflection.pushConstant[i].range,
    };
  desc.bindGroupLayouts.resize(bindgroupLayouts.size());
  for (int i = 0; i < bindgroupLayouts.size(); ++i)
    desc.bindGroupLayouts[i] = bindgroupLayouts[i].get();
  pipelineLayout = device->createPipelineLayout(desc);
}

auto PipelinePass::updateBinding(
    RenderContext* context,
    std::string const& name,
    rhi::BindingResource const& resource) noexcept -> void {
  auto iter = reflection.bindingInfo.find(name);
  if (iter == reflection.bindingInfo.end()) {
    root::print::error("RDG::Binding Name " + name + " not found");
  }
  std::vector<rhi::BindGroupEntry> set_entries = {
    rhi::BindGroupEntry{iter->second.binding, resource}};
  getBindGroup(context, iter->second.set)->updateBinding(set_entries);
}

auto PipelinePass::updateBindings(
  RenderContext* context,
  std::vector<std::pair<std::string, rhi::BindingResource>> const& bindings) noexcept -> void {
  for (auto& pair : bindings) updateBinding(context, pair.first, pair.second);
}

auto RenderPass::beginPass(RenderContext* context, gfx::Texture* target) noexcept -> rhi::RenderPassEncoder* {
  passEncoders[context->flightIdx] = context->cmdEncoder->beginRenderPass(renderPassDescriptor);
  passEncoders[context->flightIdx]->setPipeline(pipelines[context->flightIdx].get());
  prepareDispatch(context, target);
  return passEncoders[context->flightIdx].get();
}

auto RenderPass::beginPass(RenderContext* context, uint32_t width, uint32_t height) noexcept -> rhi::RenderPassEncoder* {
  passEncoders[context->flightIdx] = context->cmdEncoder->beginRenderPass(renderPassDescriptor);
  passEncoders[context->flightIdx]->setPipeline(pipelines[context->flightIdx].get());
  prepareDispatch(context, width, height);
  return passEncoders[context->flightIdx].get();
}

auto RenderPass::prepareDelegateData(RenderContext* context,
  RenderData const& renderData) noexcept
  -> RenderData::DelegateData {
  RenderData::DelegateData data;
  data.cmdEncoder = context->cmdEncoder;
  data.passEncoder.render = passEncoders[context->flightIdx].get();
  data.pipelinePass = this;
  return data;
}

auto RenderPass::generateMarker() noexcept -> void {
  marker.name = identifier;
  marker.color = { 0.764, 0.807, 0.725, 1. };
}

auto RenderPass::prepareDispatch(RenderContext* context, gfx::Texture* target) noexcept -> void {
  for (size_t i = 0; i < bindgroups.size(); ++i)
    passEncoders[context->flightIdx]->setBindGroup(
        i, bindgroups[i][context->flightIdx].get());
  passEncoders[context->flightIdx]->setViewport(
      0, 0, target->texture->width(), target->texture->height(), 0, 1);
  passEncoders[context->flightIdx]->setScissorRect(
      0, 0, target->texture->width(), target->texture->height());
}

auto RenderPass::prepareDispatch(rdg::RenderContext* context, uint32_t width, uint32_t height) noexcept -> void {
  for (size_t i = 0; i < bindgroups.size(); ++i)
    passEncoders[context->flightIdx]->setBindGroup(
        i, bindgroups[i][context->flightIdx].get());
  passEncoders[context->flightIdx]->setViewport(0, 0, width, height, 0, 1);
  passEncoders[context->flightIdx]->setScissorRect(0, 0, width, height);
}

auto RenderPass::init(gfx::ShaderModule* vertex, gfx::ShaderModule* fragment,
    std::optional<RenderPipelineDescCallback> callback) noexcept -> void {
  PipelinePass::init(std::vector<gfx::ShaderModule*>{vertex, fragment});
  rhi::Device* device = gfx::GFXContext::device;
  rhi::RenderPipelineDescriptor pipelineDesc = rhi::RenderPipelineDescriptor{
      pipelineLayout.get(),
      rhi::VertexState{// vertex shader
                       vertex->shaderModule.get(),
                       "main",
                       // vertex attribute layout
                       {}},
      rhi::PrimitiveState{rhi::PrimitiveTopology::TRIANGLE_LIST,
                          rhi::IndexFormat::UINT16_t},
      pReflection.getDepthStencilState(),
      rhi::MultisampleState{},
      rhi::FragmentState{// fragment shader
                         fragment->shaderModule.get(), "main",
                         pReflection.getColorTargetState()}};
  if (callback.has_value()) {
    callback.value()(pipelineDesc);
  }
  for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
    pipelines[i] = device->createRenderPipeline(pipelineDesc);
  }
}

auto RenderPass::init(gfx::ShaderModule* vertex,
    gfx::ShaderModule* geometry,
    gfx::ShaderModule* fragment,
    std::optional<RenderPipelineDescCallback> callback) noexcept -> void {
  PipelinePass::init(
      std::vector<gfx::ShaderModule*>{vertex, geometry, fragment});
  rhi::Device* device = gfx::GFXContext::device;
  rhi::RenderPipelineDescriptor pipelineDesc = rhi::RenderPipelineDescriptor{
      pipelineLayout.get(),
      rhi::VertexState{// vertex shader
                       vertex->shaderModule.get(),
                       "main",
                       // vertex attribute layout
                       {}},
      rhi::PrimitiveState{rhi::PrimitiveTopology::TRIANGLE_LIST,
                          rhi::IndexFormat::UINT16_t},
      pReflection.getDepthStencilState(),
      rhi::MultisampleState{},
      rhi::FragmentState{// fragment shader
                         fragment->shaderModule.get(), "main",
                         pReflection.getColorTargetState()}};
  pipelineDesc.geometry = {geometry->shaderModule.get()};
  if (callback.has_value()) {
    callback.value()(pipelineDesc);
  }
  for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
    pipelines[i] = device->createRenderPipeline(pipelineDesc);
  }
}

auto FullScreenPass::dispatchFullScreen(rdg::RenderContext* context) noexcept -> void {
    passEncoders[context->flightIdx]->draw(3, 1, 0, 0);
}

gfx::ShaderModule* FullScreenPass::fullscreen_vertex;

auto FullScreenPass::init(gfx::ShaderModule* fragment) noexcept -> void {
  if (FullScreenPass::fullscreen_vertex == nullptr) {
    //auto vert = GFX::GFXManager::get()->registerShaderModuleResource(
    //    (engine_path +
    //     "/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/"
    //     "fullscreen_pass_vert.spv")
    //        .c_str(),
    //    {nullptr, RHI::ShaderStages::VERTEX});
    //FullScreenPass::fullscreen_vertex =
    //    Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert);
  }
  RenderPass::init(FullScreenPass::fullscreen_vertex, fragment);
}

auto FullScreenPass::generateMarker() noexcept -> void {
  marker.name = identifier;
  marker.color = { 0.682, 0.764, 0.752, 1. };
}

auto ComputePass::beginPass(rdg::RenderContext* context) noexcept -> rhi::ComputePassEncoder* {
  passEncoders[context->flightIdx] = context->cmdEncoder->beginComputePass(rhi::ComputePassDescriptor{});
  passEncoders[context->flightIdx]->setPipeline(pipelines[context->flightIdx].get());
  prepareDispatch(context);
  return passEncoders[context->flightIdx].get();
}

auto ComputePass::generateMarker() noexcept -> void {
  marker.name = identifier;
  marker.color = { 0.6, 0.721, 0.780, 1. };
}

auto ComputePass::prepareDispatch(rdg::RenderContext* context) noexcept -> void {
  for (size_t i = 0; i < bindgroups.size(); ++i)
    passEncoders[context->flightIdx]->setBindGroup(
      i, bindgroups[i][context->flightIdx].get());
}

auto ComputePass::init(gfx::ShaderModule* comp) noexcept -> void {
  PipelinePass::init({comp});
  rhi::Device* device = gfx::GFXContext::device;
  for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
    pipelines[i] = device->createComputePipeline(rhi::ComputePipelineDescriptor{
        pipelineLayout.get(), {comp->shaderModule.get(), "main"}});
  }
}

auto RayTracingPass::beginPass(rdg::RenderContext* context) noexcept -> rhi::RayTracingPassEncoder* {
  passEncoders[context->flightIdx] = context->cmdEncoder->beginRayTracingPass(rhi::RayTracingPassDescriptor{});
  passEncoders[context->flightIdx]->setPipeline(pipelines[context->flightIdx].get());
  prepareDispatch(context);
  return passEncoders[context->flightIdx].get();
}

auto RayTracingPass::generateMarker() noexcept -> void {
  marker.name = identifier;
  marker.color = { 0.517, 0.674, 0.807, 1. };
}

auto RayTracingPass::prepareDispatch(rdg::RenderContext* context) noexcept -> void {
  for (size_t i = 0; i < bindgroups.size(); ++i)
    passEncoders[context->flightIdx]->setBindGroup(
      i, bindgroups[i][context->flightIdx].get());
}
//
//auto Graph::build() noexcept -> bool {
//  renderData.graph = this;
//  // Flatten the graph
//  std::optional<std::vector<size_t>> flat =
//      flatten_bfs(dag, passNameList[output_pass]);
//  if (!flat.has_value()) return false;
//  flattenedPasses = flat.value();
//
//  // Find the resource
//  for (size_t i = 0; i < flattenedPasses.size(); ++i) {
//    // devirtualize internal resources
//    for (auto& internal :
//         passes[flattenedPasses[i]]->pReflection.internalResources) {
//      if (internal.second.type == ResourceInfo::Type::Texture) {
//        size_t rid = resourceID++;
//        textureResources[rid] = std::make_unique<TextureResource>();
//        Core::GUID guid =
//            Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
//        textureResources[rid]->desc =
//            toTextureDescriptor(internal.second.info.texture, standardSize);
//        textureResources[rid]->name =
//            "RDG::" + passes[flattenedPasses[i]]->identifier +
//            "::" + internal.first;
//        textureResources[rid]->cosumeHistories.push_back(
//            {flattenedPasses[i],
//             internal.second.info.texture.consumeHistories});
//        internal.second.devirtualizeID = rid;
//      } else if (internal.second.type == ResourceInfo::Type::Buffer) {
//        size_t rid = resourceID++;
//        bufferResources[rid] = std::make_unique<BufferResource>();
//        Core::GUID guid =
//            Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
//        bufferResources[rid]->desc =
//            toBufferDescriptor(internal.second.info.buffer);
//        bufferResources[rid]->cosumeHistories.push_back(
//            {flattenedPasses[i], internal.second.info.buffer.consumeHistories});
//        internal.second.devirtualizeID = rid;
//      }
//    }
//    for (auto& internal :
//         passes[flattenedPasses[i]]->pReflection.outputResources) {
//      if (internal.second.type == ResourceInfo::Type::Texture) {
//        size_t rid = resourceID++;
//        textureResources[rid] = std::make_unique<TextureResource>();
//        Core::GUID guid =
//            Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
//        textureResources[rid]->desc =
//            toTextureDescriptor(internal.second.info.texture, standardSize);
//        textureResources[rid]->name =
//            "RDG::" + passes[flattenedPasses[i]]->identifier +
//            "::" + internal.first;
//        textureResources[rid]->cosumeHistories.push_back(
//            {flattenedPasses[i],
//             internal.second.info.texture.consumeHistories});
//        internal.second.devirtualizeID = rid;
//        if (internal.second.info.texture.reference != nullptr)
//          textureResources[rid]->texture =
//              internal.second.info.texture.reference;
//      } else if (internal.second.type == ResourceInfo::Type::Buffer) {
//        size_t rid = resourceID++;
//        bufferResources[rid] = std::make_unique<BufferResource>();
//        Core::GUID guid =
//            Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
//        bufferResources[rid]->desc =
//            toBufferDescriptor(internal.second.info.buffer);
//        bufferResources[rid]->cosumeHistories.push_back(
//            {flattenedPasses[i], internal.second.info.buffer.consumeHistories});
//        internal.second.devirtualizeID = rid;
//        if (internal.second.info.buffer.reference != nullptr)
//          bufferResources[rid]->buffer =
//              internal.second.info.buffer.reference;
//      }
//    }
//    // devirtualize input resources
//    for (auto& internal :
//         passes[flattenedPasses[i]]->pReflection.inputResources) {
//      if (internal.second.type == ResourceInfo::Type::Texture) {
//        if (!internal.second.info.texture.reference) {
//          size_t rid = internal.second.prev->devirtualizeID;
//          internal.second.devirtualizeID = rid;
//          RHI::TextureDescriptor desc =
//              toTextureDescriptor(internal.second.info.texture, standardSize);
//          textureResources[rid]->desc.usage |= desc.usage;
//          textureResources[rid]->cosumeHistories.push_back(
//              {flattenedPasses[i],
//               internal.second.info.texture.consumeHistories});
//        } else {
//          size_t rid = resourceID++;
//          textureResources[rid] = std::make_unique<TextureResource>();
//          textureResources[rid]->desc =
//              toTextureDescriptor(internal.second.info.texture, standardSize);
//          textureResources[rid]->cosumeHistories.push_back(
//              {flattenedPasses[i],
//               internal.second.info.texture.consumeHistories});
//          internal.second.devirtualizeID = rid;
//          textureResources[rid]->texture =
//              internal.second.info.texture.reference;
//        }
//      } else if (internal.second.type == ResourceInfo::Type::Buffer) {
//        size_t rid = internal.second.prev->devirtualizeID;
//        internal.second.devirtualizeID = rid;
//        RHI::BufferDescriptor desc =
//            toBufferDescriptor(internal.second.info.buffer);
//        bufferResources[rid]->desc.usage |= desc.usage;
//        bufferResources[rid]->cosumeHistories.push_back(
//            {flattenedPasses[i], internal.second.info.buffer.consumeHistories});
//      }
//    }
//    // devirtualize input-output resources
//    for (auto& internal :
//         passes[flattenedPasses[i]]->pReflection.inputOutputResources) {
//      if (internal.second.type == ResourceInfo::Type::Texture) {
//        if (internal.second.prev == nullptr) {
//          Core::LogManager::Error(std::format(
//              "RDG::Graph::build() failed, input-output resource \"{0}\" in "
//              "pass \"{1}\" has no source.",
//              internal.first, passes[flattenedPasses[i]]->identifier));
//        }
//        size_t rid = internal.second.prev->devirtualizeID;
//        internal.second.devirtualizeID = rid;
//        RHI::TextureDescriptor desc =
//            toTextureDescriptor(internal.second.info.texture, standardSize);
//        textureResources[rid]->desc.usage |= desc.usage;
//        textureResources[rid]->cosumeHistories.push_back(
//            {flattenedPasses[i],
//             internal.second.info.texture.consumeHistories});
//      } else if (internal.second.type == ResourceInfo::Type::Buffer) {
//        size_t rid = internal.second.prev->devirtualizeID;
//        internal.second.devirtualizeID = rid;
//        RHI::BufferDescriptor desc =
//            toBufferDescriptor(internal.second.info.buffer);
//        bufferResources[rid]->desc.usage |= desc.usage;
//        bufferResources[rid]->cosumeHistories.push_back(
//            {flattenedPasses[i], internal.second.info.buffer.consumeHistories});
//      }
//    }
//    // Deal with relative to another tex case
//    for (auto& internal :
//         passes[flattenedPasses[i]]->pReflection.internalResources) {
//      if (internal.second.type == ResourceInfo::Type::Texture) {
//        if (internal.second.info.texture.sizeDef ==
//            TextureInfo::SizeDefine::RelativeToAnotherTex) {
//          auto const& refResourceInfo =
//              passes[flattenedPasses[i]]->pReflection.getResourceInfo(
//                  internal.second.info.texture.sizeRefName);
//          textureResources[internal.second.devirtualizeID]->desc.size =
//              textureResources[refResourceInfo->devirtualizeID]->desc.size;
//        }
//      }
//    }
//    // devirtualize output resources
//    for (auto& internal :
//         passes[flattenedPasses[i]]->pReflection.outputResources) {
//      if (internal.second.type == ResourceInfo::Type::Texture) {
//        if (internal.second.info.texture.sizeDef ==
//            TextureInfo::SizeDefine::RelativeToAnotherTex) {
//          auto const& refResourceInfo =
//              passes[flattenedPasses[i]]->pReflection.getResourceInfo(
//                  internal.second.info.texture.sizeRefName);
//          textureResources[internal.second.devirtualizeID]->desc.size =
//              textureResources[refResourceInfo->devirtualizeID]->desc.size;
//        }
//      }
//    }
//  }
//
//  // devirtualize all the results
//  for (auto& res : textureResources) {
//    if (res.second->desc.mipLevelCount == uint32_t(-1)) {
//      res.second->desc.mipLevelCount =
//          std::log2(std::max(res.second->desc.size.width,
//                             res.second->desc.size.height)) +
//          1;
//    }
//
//    RDG::TextureInfo::ConsumeEntry final_consume =
//        RDG::TextureInfo::ConsumeEntry{
//            RDG::TextureInfo::ConsumeType::TextureBinding}
//            .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT)
//            .setLayout(RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL)
//            .setAccess(uint32_t(RHI::AccessFlagBits::SHADER_READ_BIT))
//            .setSubresource(0, res.second->desc.mipLevelCount, 0,
//                            res.second->desc.arrayLayerCount);
//    res.second->cosumeHistories.push_back({size_t(-1), {final_consume}});
//
//    if (res.second->texture == nullptr) {
//      res.second->guid =
//          Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
//      GFX::GFXManager::get()->registerTextureResource(res.second->guid,
//                                                      res.second->desc);
//      res.second->texture =
//          Core::ResourceManager::get()->getResource<GFX::Texture>(
//              res.second->guid);
//      res.second->texture->texture->setName(res.second->name);
//    }
//  }
//
//  for (auto& res : bufferResources) {
//    if (res.second->buffer == nullptr) {
//      res.second->guid =
//          Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
//      GFX::GFXManager::get()->registerBufferResource(res.second->guid,
//                                                     res.second->desc);
//      res.second->buffer =
//          Core::ResourceManager::get()->getResource<GFX::Buffer>(
//              res.second->guid);
//    }
//  }
//  // create barriers
//  generateTextureBarriers();
//  generateBufferBarriers();
//}

}