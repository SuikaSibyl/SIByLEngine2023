#include "../Public/SE.RDG-Main.hpp"

namespace SIByL::RDG {
GFX::ShaderModule* FullScreenPass::fullscreen_vertex;

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
auto PassReflection::getDepthStencilState() noexcept -> RHI::DepthStencilState {
  std::vector<std::unordered_map<std::string, ResourceInfo>*> pools =
      std::vector<std::unordered_map<std::string, ResourceInfo>*>{
          &inputResources, &outputResources, &inputOutputResources,
          &internalResources};
  for (auto& pool : pools) {
    for (auto& pair : *pool) {
      if (pair.second.type != ResourceInfo::Type::Texture) continue;
      if (RHI::hasDepthBit(pair.second.info.texture.format)) {
        TextureInfo::ConsumeEntry entry;
        for (auto const& ch : pair.second.info.texture.consumeHistories) {
          if (ch.type == TextureInfo::ConsumeType::DepthStencilAttachment) {
            entry = ch;
            break;
          }
        }
        return RHI::DepthStencilState{pair.second.info.texture.format,
                                      entry.depthWrite, entry.depthCmp};
      }
    }
  }
  return RHI::DepthStencilState{};
}
auto PassReflection::getColorTargetState() noexcept
    -> std::vector<RHI::ColorTargetState> {
  std::vector<RHI::ColorTargetState> cts;
  std::vector<std::unordered_map<std::string, ResourceInfo>*> pools =
      std::vector<std::unordered_map<std::string, ResourceInfo>*>{
          &inputResources, &outputResources, &inputOutputResources,
          &internalResources};
  for (auto& pool : pools) {
    for (auto& pair : *pool) {
      if (pair.second.type != ResourceInfo::Type::Texture) continue;
      if (!RHI::hasDepthBit(pair.second.info.texture.format)) {
        for (auto const& history : pair.second.info.texture.consumeHistories) {
          if (history.type != TextureInfo::ConsumeType::ColorAttachment)
            continue;
          cts.resize(history.attachLoc + 1);
          cts[history.attachLoc] =
              RHI::ColorTargetState{pair.second.info.texture.format};
        }
      }
    }
  }
  return cts;
}

auto Graph::addPass(std::unique_ptr<Pass>&& pass,
                    std::string const& identifier) noexcept -> void {
  passNameList[identifier] = passID;
  pass->identifier = identifier;
  pass->subgraphStack = subgraphStack;
  pass->generateMarker();
  passes[passID++] = std::move(pass);
}

auto Graph::addSubgraph(std::unique_ptr<Subgraph>&& subgraph,
                        std::string const& identifier) noexcept -> void {
  uint32_t id = subgraphID++;
  subgraphNameList[id] = identifier;
  subgraph->identifier = identifier;
  subgraphsAlias[identifier] = subgraph->alias();
  subgraphStack.push_back(id);
  subgraph->onRegister(this);
  subgraphStack.pop_back();
  subgraph->generateMarker();
  subgraphs[identifier] = std::move(subgraph);
}

auto Graph::addEdge(std::string const& _src_pass,
                    std::string const& _src_resource,
                    std::string const& _dst_pass,
                    std::string const& _dst_resource) noexcept -> void {
  std::string src_pass = _src_pass;
  std::string dst_pass = _dst_pass;
  std::string src_resource = _src_resource;
  std::string dst_resource = _dst_resource;
  decodeAlias(src_pass, src_resource);
  decodeAlias(dst_pass, dst_resource);

  dag.addEdge(passNameList[src_pass], passNameList[dst_pass]);
  auto* dst_res =
      passes[passNameList[dst_pass]]->pReflection.getResourceInfo(dst_resource);
  auto* src_res =
      passes[passNameList[src_pass]]->pReflection.getResourceInfo(src_resource);
  if (dst_res == nullptr) {
    Core::LogManager::Error(
        std::format("RDG::Graph::addEdge Failed to find dst resource \"{0}\" "
                    "in pass \"{1}\"",
                    dst_resource, dst_pass));
    return;
  }
  if (src_res == nullptr) {
    Core::LogManager::Error(
        std::format("RDG::Graph::addEdge Failed to find src resource \"{0}\" "
                    "in pass \"{1}\"",
                    src_resource, src_pass));
    return;
  }
  dst_res->prev = src_res;
}

auto Graph::addEdge(std::string const& src_pass,
                    std::string const& dst_pass) noexcept -> void {
  dag.addEdge(passNameList[src_pass], passNameList[dst_pass]);
}

auto Graph::decodeAlias(std::string& pass, std::string& resource) noexcept
    -> void {
  auto findPass = subgraphsAlias.find(pass);
  if (findPass != subgraphsAlias.end()) {
    auto findResource = findPass->second.dict.find(resource);
    if (findResource != findPass->second.dict.end()) {
      pass = findResource->second.pass;
      resource = findResource->second.resource;
    }
  }
}

auto Graph::setExternal(std::string const& _pass, std::string const& _resource,
                        GFX::Texture* tex) noexcept -> void {
  std::string pass = _pass, res = _resource;
  decodeAlias(pass, res);
  passes[passNameList[pass]]
      ->pReflection.getResourceInfo(res)
      ->info.texture.reference = tex;
}

auto Graph::markOutput(std::string const& _pass,
                       std::string const& _output) noexcept -> void {
  std::string pass = _pass, output = _output;
  decodeAlias(pass, output);
  output_pass = pass;
  output_resource = output;

  auto const& iter_id = passNameList.find(output_pass);
  if (iter_id == passNameList.end()) return;
  size_t id = iter_id->second;
  auto const& iter_pass = passes.find(id);
  if (iter_pass == passes.end()) return;
  // enable color attachment for editor debug draw
  iter_pass->second->pReflection.getResourceInfo(output_resource)
      ->info.texture.usages |= (uint32_t)(RHI::TextureUsage::COLOR_ATTACHMENT);
}

auto Pass::init() noexcept -> void { pReflection = this->reflect(); }

auto PipelinePass::init(std::vector<GFX::ShaderModule*> shaderModules) noexcept
    -> void {
  Pass::init();
  RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
  // create pipeline reflection
  for (auto& sm : shaderModules) reflection = reflection + sm->reflection;
  // create bindgroup layouts
  bindgroupLayouts.resize(reflection.bindings.size());
  for (int i = 0; i < reflection.bindings.size(); ++i) {
    RHI::BindGroupLayoutDescriptor desc =
        GFX::ShaderReflection::toBindGroupLayoutDescriptor(
            reflection.bindings[i]);
    bindgroupLayouts[i] = device->createBindGroupLayout(desc);
  }
  // create bindgroups
  bindgroups.resize(reflection.bindings.size());
  for (size_t i = 0; i < reflection.bindings.size(); ++i) {
    for (size_t j = 0; j < MULTIFRAME_FLIGHTS_COUNT; ++j) {
      bindgroups[i][j] = device->createBindGroup(RHI::BindGroupDescriptor{
          bindgroupLayouts[i].get(),
          std::vector<RHI::BindGroupEntry>{},
      });
    }
  }
  // create pipelineLayout
  RHI::PipelineLayoutDescriptor desc = {};
  desc.pushConstants.resize(reflection.pushConstant.size());
  for (int i = 0; i < reflection.pushConstant.size(); ++i)
    desc.pushConstants[i] = RHI::PushConstantEntry{
        reflection.pushConstant[i].stages,
        reflection.pushConstant[i].offset,
        reflection.pushConstant[i].range,
    };
  desc.bindGroupLayouts.resize(bindgroupLayouts.size());
  for (int i = 0; i < bindgroupLayouts.size(); ++i)
    desc.bindGroupLayouts[i] = bindgroupLayouts[i].get();
  pipelineLayout = device->createPipelineLayout(desc);
}

auto RenderPass::init(GFX::ShaderModule* vertex,
                      GFX::ShaderModule* fragment) noexcept -> void {
  PipelinePass::init(std::vector<GFX::ShaderModule*>{vertex, fragment});
  RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
  for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
    pipelines[i] = device->createRenderPipeline(RHI::RenderPipelineDescriptor{
        pipelineLayout.get(),
        RHI::VertexState{// vertex shader
                         vertex->shaderModule.get(),
                         "main",
                         // vertex attribute layout
                         {}},
        RHI::PrimitiveState{RHI::PrimitiveTopology::TRIANGLE_LIST,
                            RHI::IndexFormat::UINT16_t},
        pReflection.getDepthStencilState(), RHI::MultisampleState{},
        RHI::FragmentState{// fragment shader
                           fragment->shaderModule.get(), "main",
                           pReflection.getColorTargetState()}});
  }
}

auto FullScreenPass::init(GFX::ShaderModule* fragment) noexcept -> void {
  if (FullScreenPass::fullscreen_vertex == nullptr) {
    auto vert = GFX::GFXManager::get()->registerShaderModuleResource(
        "../Engine/Binaries/Runtime/spirv/SRenderer/rasterizer/fullscreen_pass/"
        "fullscreen_pass_vert.spv",
        {nullptr, RHI::ShaderStages::VERTEX});
    FullScreenPass::fullscreen_vertex =
        Core::ResourceManager::get()->getResource<GFX::ShaderModule>(vert);
  }
  RenderPass::init(FullScreenPass::fullscreen_vertex, fragment);
}

auto ComputePass::init(GFX::ShaderModule* comp) noexcept -> void {
  PipelinePass::init({comp});
  RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
  for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
    pipelines[i] = device->createComputePipeline(RHI::ComputePipelineDescriptor{
        pipelineLayout.get(), {comp->shaderModule.get(), "main"}});
  }
}

auto collectAllShaders(GFX::SBTsDescriptor const& sbt) noexcept
    -> std::vector<GFX::ShaderModule*> {
  std::vector<GFX::ShaderModule*> sms;
  sms.push_back(sbt.rgenSBT.rgenRecord.rayGenShader);
  for (auto const& entry : sbt.hitGroupSBT.hitGroupRecords) {
    if (entry.anyHitShader) sms.push_back(entry.anyHitShader);
    if (entry.closetHitShader) sms.push_back(entry.closetHitShader);
    if (entry.intersectionShader) sms.push_back(entry.intersectionShader);
  }
  for (auto const& entry : sbt.missSBT.rmissRecords)
    sms.push_back(entry.missShader);
  for (auto const& entry : sbt.callableSBT.callableRecords)
    sms.push_back(entry.callableShader);
  return sms;
}

auto RayTracingPass::init(GFX::SBTsDescriptor const& sbt,
                          uint32_t max_depth) noexcept -> void {
  std::vector<GFX::ShaderModule*> shaderModules = collectAllShaders(sbt);
  PipelinePass::init(shaderModules);
  RHI::Device* device = GFX::GFXManager::get()->rhiLayer->getDevice();
  for (int i = 0; i < MULTIFRAME_FLIGHTS_COUNT; ++i) {
    pipelines[i] =
        device->createRayTracingPipeline(RHI::RayTracingPipelineDescriptor{
            pipelineLayout.get(), max_depth, RHI::SBTsDescriptor(sbt)});
  }
}

auto Graph::execute(RHI::CommandEncoder* encoder) noexcept -> void {
  renderData.graph = this;
  RenderContext renderContext;
  renderContext.cmdEncoder = encoder;
  renderContext.flightIdx = GFX::GFXManager::get()
                                ->rhiLayer->getMultiFrameFlights()
                                ->getFlightIndex();

  std::vector<size_t> marker_stack;

  for (size_t pass_id : flattenedPasses) {
    auto* pass = passes[pass_id].get();
    renderData.pass = pass;
    // insert barriers
    for (auto const& barriers : barriers[pass_id]) {
      renderContext.cmdEncoder->pipelineBarrier(barriers);
    }

    {  // deal with subgroup markers
      if (marker_stack.empty() && !pass->subgraphStack.empty()) {
        for (auto const& subgraph_id : pass->subgraphStack) {
          encoder->beginDebugUtilsLabelEXT(
              subgraphs[subgraphNameList[subgraph_id]]->marker);
          marker_stack.push_back(subgraph_id);
        }
      } else if (!marker_stack.empty() && pass->subgraphStack.empty()) {
        while (marker_stack.size() > 0) {
          marker_stack.pop_back();
          encoder->endDebugUtilsLabelEXT();
        }
      } else if (!marker_stack.empty() &&
                 marker_stack.back() != pass->subgraphStack.back()) {
        size_t offset = 0;
        while (offset < marker_stack.size() &&
               offset < pass->subgraphStack.size() &&
               marker_stack[offset] == pass->subgraphStack[offset]) {
          offset++;
        }
        while (marker_stack.size() > offset) {
          marker_stack.pop_back();
          encoder->endDebugUtilsLabelEXT();
        }
        for (size_t i = offset; i < pass->subgraphStack.size(); ++i) {
          encoder->beginDebugUtilsLabelEXT(
              subgraphs[subgraphNameList[pass->subgraphStack[i]]]->marker);
          marker_stack.push_back(pass->subgraphStack[i]);
        }
      }
    }

    encoder->beginDebugUtilsLabelEXT(pass->marker);
    pass->execute(&renderContext, renderData);
    encoder->endDebugUtilsLabelEXT();
  }

  while (marker_stack.size() > 0) {
    marker_stack.pop_back();
    encoder->endDebugUtilsLabelEXT();
  }

  {
    // insert barriers
    for (auto const& barriers : barriers[size_t(-1)])
      renderContext.cmdEncoder->pipelineBarrier(barriers);
  }
}

auto tryMerge(RHI::ImageSubresourceRange const& x,
              RHI::ImageSubresourceRange const& y) noexcept
    -> std::optional<RHI::ImageSubresourceRange> {
  RHI::ImageSubresourceRange range;
  if (x.aspectMask != y.aspectMask) return std::nullopt;
  if (x.baseArrayLayer == y.baseArrayLayer && x.layerCount == y.layerCount) {
    if (x.baseMipLevel + x.levelCount == y.baseMipLevel) {
      return RHI::ImageSubresourceRange{x.aspectMask, x.baseMipLevel,
                                        x.levelCount + y.levelCount,
                                        x.baseArrayLayer, x.layerCount};
    } else if (y.baseMipLevel + y.levelCount == x.baseMipLevel) {
      return RHI::ImageSubresourceRange{x.aspectMask, y.baseMipLevel,
                                        x.levelCount + y.levelCount,
                                        x.baseArrayLayer, x.layerCount};
    } else
      return std::nullopt;
  } else if (x.baseMipLevel == y.baseMipLevel && x.levelCount == y.levelCount) {
    if (x.baseArrayLayer + x.layerCount == y.baseArrayLayer) {
      return RHI::ImageSubresourceRange{x.aspectMask, x.baseMipLevel,
                                        x.levelCount, x.baseArrayLayer,
                                        x.layerCount + y.layerCount};
    } else if (y.baseArrayLayer + y.layerCount == x.baseArrayLayer) {
      return RHI::ImageSubresourceRange{x.aspectMask, x.baseMipLevel,
                                        x.levelCount, y.baseArrayLayer,
                                        x.layerCount + y.layerCount};
    } else
      return std::nullopt;
  } else
    return std::nullopt;
}

auto tryMergeBarriers(std::vector<RHI::BarrierDescriptor>& barriers) noexcept
    -> void {
  // try merge macro barriers
  while (true) {
    bool should_continue = false;
    for (int i = 0; i < barriers.size(); ++i) {
      if (should_continue) break;
      for (int j = i + 1; j < barriers.size(); ++j) {
        if (should_continue) break;
        if ((barriers[i].srcStageMask == barriers[j].srcStageMask) &&
            (barriers[i].dstStageMask == barriers[j].dstStageMask)) {
          barriers[i].memoryBarriers.insert(barriers[i].memoryBarriers.end(),
                                            barriers[j].memoryBarriers.begin(),
                                            barriers[j].memoryBarriers.end());
          barriers[i].bufferMemoryBarriers.insert(
              barriers[i].bufferMemoryBarriers.end(),
              barriers[j].bufferMemoryBarriers.begin(),
              barriers[j].bufferMemoryBarriers.end());
          barriers[i].textureMemoryBarriers.insert(
              barriers[i].textureMemoryBarriers.end(),
              barriers[j].textureMemoryBarriers.begin(),
              barriers[j].textureMemoryBarriers.end());
          barriers.erase(barriers.begin() + j);
          should_continue = true;
        }
      }
    }
    if (should_continue) continue;
    break;
  }
  // try merge texture sub barriers
  for (int k = 0; k < barriers.size(); ++k) {
    auto& barrier = barriers[k];
    auto& textureBarriers = barrier.textureMemoryBarriers;
    while (true) {
      bool should_continue = false;
      for (int i = 0; i < textureBarriers.size(); ++i) {
        if (should_continue) break;
        for (int j = i + 1; j < textureBarriers.size(); ++j) {
          if (should_continue) break;
          if ((textureBarriers[i].texture == textureBarriers[j].texture) &&
              (textureBarriers[i].srcAccessMask ==
               textureBarriers[j].srcAccessMask) &&
              (textureBarriers[i].dstAccessMask ==
               textureBarriers[j].dstAccessMask) &&
              (textureBarriers[i].oldLayout == textureBarriers[j].oldLayout) &&
              (textureBarriers[i].newLayout == textureBarriers[j].newLayout)) {
            auto merged = tryMerge(textureBarriers[i].subresourceRange,
                                   textureBarriers[j].subresourceRange);
            if (merged.has_value()) {
              textureBarriers[i].subresourceRange = merged.value();
              textureBarriers.erase(textureBarriers.begin() + j);
              should_continue = true;
            }
          }
        }
      }
      if (should_continue) continue;
      break;
    }
  }
}

auto Graph::build() noexcept -> bool {
  renderData.graph = this;
  // Flatten the graph
  std::optional<std::vector<size_t>> flat =
      flatten_bfs(dag, passNameList[output_pass]);
  if (!flat.has_value()) return false;
  flattenedPasses = flat.value();

  // Find the resource
  for (size_t i = 0; i < flattenedPasses.size(); ++i) {
    // devirtualize internal resources
    for (auto& internal :
         passes[flattenedPasses[i]]->pReflection.internalResources) {
      if (internal.second.type == ResourceInfo::Type::Texture) {
        size_t rid = resourceID++;
        textureResources[rid] = std::make_unique<TextureResource>();
        Core::GUID guid =
            Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
        textureResources[rid]->desc =
            toTextureDescriptor(internal.second.info.texture, standardSize);
        textureResources[rid]->cosumeHistories.push_back(
            {flattenedPasses[i],
             internal.second.info.texture.consumeHistories});
        internal.second.devirtualizeID = rid;
      } else if (internal.second.type == ResourceInfo::Type::Buffer) {
        size_t rid = resourceID++;
        bufferResources[rid] = std::make_unique<BufferResource>();
        Core::GUID guid =
            Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
        bufferResources[rid]->desc =
            toBufferDescriptor(internal.second.info.buffer);
        bufferResources[rid]->cosumeHistories.push_back(
            {flattenedPasses[i], internal.second.info.buffer.consumeHistories});
        internal.second.devirtualizeID = rid;
      }
    }
    for (auto& internal :
         passes[flattenedPasses[i]]->pReflection.outputResources) {
      if (internal.second.type == ResourceInfo::Type::Texture) {
        size_t rid = resourceID++;
        textureResources[rid] = std::make_unique<TextureResource>();
        Core::GUID guid =
            Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
        textureResources[rid]->desc =
            toTextureDescriptor(internal.second.info.texture, standardSize);
        textureResources[rid]->cosumeHistories.push_back(
            {flattenedPasses[i],
             internal.second.info.texture.consumeHistories});
        internal.second.devirtualizeID = rid;
        if (internal.second.info.texture.reference != nullptr)
          textureResources[rid]->texture =
              internal.second.info.texture.reference;
      } else if (internal.second.type == ResourceInfo::Type::Buffer) {
        size_t rid = resourceID++;
        bufferResources[rid] = std::make_unique<BufferResource>();
        Core::GUID guid =
            Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
        bufferResources[rid]->desc =
            toBufferDescriptor(internal.second.info.buffer);
        bufferResources[rid]->cosumeHistories.push_back(
            {flattenedPasses[i], internal.second.info.buffer.consumeHistories});
        internal.second.devirtualizeID = rid;
      }
    }
    // devirtualize input resources
    for (auto& internal :
         passes[flattenedPasses[i]]->pReflection.inputResources) {
      if (internal.second.type == ResourceInfo::Type::Texture) {
        if (!internal.second.info.texture.reference) {
          size_t rid = internal.second.prev->devirtualizeID;
          internal.second.devirtualizeID = rid;
          RHI::TextureDescriptor desc =
              toTextureDescriptor(internal.second.info.texture, standardSize);
          textureResources[rid]->desc.usage |= desc.usage;
          textureResources[rid]->cosumeHistories.push_back(
              {flattenedPasses[i],
               internal.second.info.texture.consumeHistories});
        } else {
          size_t rid = resourceID++;
          textureResources[rid] = std::make_unique<TextureResource>();
          textureResources[rid]->desc =
              toTextureDescriptor(internal.second.info.texture, standardSize);
          textureResources[rid]->cosumeHistories.push_back(
              {flattenedPasses[i],
               internal.second.info.texture.consumeHistories});
          internal.second.devirtualizeID = rid;
          textureResources[rid]->texture =
              internal.second.info.texture.reference;
        }
      } else if (internal.second.type == ResourceInfo::Type::Buffer) {
        size_t rid = internal.second.prev->devirtualizeID;
        internal.second.devirtualizeID = rid;
        bufferResources[rid] = std::make_unique<BufferResource>();
        Core::GUID guid =
            Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
        RHI::BufferDescriptor desc =
            toBufferDescriptor(internal.second.info.buffer);
        bufferResources[rid]->desc.usage |= desc.usage;
        bufferResources[rid]->cosumeHistories.push_back(
            {flattenedPasses[i], internal.second.info.buffer.consumeHistories});
      }
    }
    // devirtualize input-output resources
    for (auto& internal :
         passes[flattenedPasses[i]]->pReflection.inputOutputResources) {
      if (internal.second.type == ResourceInfo::Type::Texture) {
        if (internal.second.prev == nullptr) {
          Core::LogManager::Error(std::format(
              "RDG::Graph::build() failed, input-output resource \"{0}\" in "
              "pass \"{1}\" has no source.",
              internal.first, passes[flattenedPasses[i]]->identifier));
        }
        size_t rid = internal.second.prev->devirtualizeID;
        internal.second.devirtualizeID = rid;
        RHI::TextureDescriptor desc =
            toTextureDescriptor(internal.second.info.texture, standardSize);
        textureResources[rid]->desc.usage |= desc.usage;
        textureResources[rid]->cosumeHistories.push_back(
            {flattenedPasses[i],
             internal.second.info.texture.consumeHistories});
      } else if (internal.second.type == ResourceInfo::Type::Buffer) {
        size_t rid = internal.second.prev->devirtualizeID;
        internal.second.devirtualizeID = rid;
        bufferResources[rid] = std::make_unique<BufferResource>();
        Core::GUID guid =
            Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
        RHI::BufferDescriptor desc =
            toBufferDescriptor(internal.second.info.buffer);
        bufferResources[rid]->desc.usage |= desc.usage;
        bufferResources[rid]->cosumeHistories.push_back(
            {flattenedPasses[i], internal.second.info.buffer.consumeHistories});
      }
    }
    // Deal with relative to another tex case
    for (auto& internal :
         passes[flattenedPasses[i]]->pReflection.internalResources) {
      if (internal.second.type == ResourceInfo::Type::Texture) {
        if (internal.second.info.texture.sizeDef ==
            TextureInfo::SizeDefine::RelativeToAnotherTex) {
          auto const& refResourceInfo =
              passes[flattenedPasses[i]]->pReflection.getResourceInfo(
                  internal.second.info.texture.sizeRefName);
          textureResources[internal.second.devirtualizeID]->desc.size =
              textureResources[refResourceInfo->devirtualizeID]->desc.size;
        }
      }
    }
    // devirtualize output resources
    for (auto& internal :
         passes[flattenedPasses[i]]->pReflection.outputResources) {
      if (internal.second.type == ResourceInfo::Type::Texture) {
        if (internal.second.info.texture.sizeDef ==
            TextureInfo::SizeDefine::RelativeToAnotherTex) {
          auto const& refResourceInfo =
              passes[flattenedPasses[i]]->pReflection.getResourceInfo(
                  internal.second.info.texture.sizeRefName);
          textureResources[internal.second.devirtualizeID]->desc.size =
              textureResources[refResourceInfo->devirtualizeID]->desc.size;
        }
      }
    }
  }

  // devirtualize all the results
  for (auto& res : textureResources) {
    if (res.second->desc.mipLevelCount == uint32_t(-1)) {
      res.second->desc.mipLevelCount =
          std::log2(std::max(res.second->desc.size.width,
                             res.second->desc.size.height)) +
          1;
    }

    RDG::TextureInfo::ConsumeEntry final_consume =
        RDG::TextureInfo::ConsumeEntry{
            RDG::TextureInfo::ConsumeType::TextureBinding}
            .addStage((uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT)
            .setLayout(RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL)
            .setAccess(uint32_t(RHI::AccessFlagBits::SHADER_READ_BIT))
            .setSubresource(0, res.second->desc.mipLevelCount, 0,
                            res.second->desc.size.depthOrArrayLayers);
    res.second->cosumeHistories.push_back({size_t(-1), {final_consume}});

    if (res.second->texture == nullptr) {
      res.second->guid =
          Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
      GFX::GFXManager::get()->registerTextureResource(res.second->guid,
                                                      res.second->desc);
      res.second->texture =
          Core::ResourceManager::get()->getResource<GFX::Texture>(
              res.second->guid);
    }
  }

  for (auto& res : bufferResources) {
    if (res.second->buffer == nullptr) {
      res.second->guid =
          Core::ResourceManager::get()->requestRuntimeGUID<GFX::Buffer>();
      GFX::GFXManager::get()->registerBufferResource(res.second->guid,
                                                     res.second->desc);
      res.second->buffer =
          Core::ResourceManager::get()->getResource<GFX::Buffer>(
              res.second->guid);
    }
  }
  // create barriers
  generateTextureBarriers();
  generateBufferBarriers();
}

auto Graph::generateBufferBarriers() noexcept -> void {
  for (auto& res : bufferResources) {
    if (res.second->cosumeHistories.size() == 0) continue;

    auto prev_ch = res.second->cosumeHistories.back().entries[0];
    auto curr_ch = prev_ch;

    for (auto const& hentry : res.second->cosumeHistories) {
      curr_ch = hentry.entries[0];
      RHI::BarrierDescriptor desc =
          RHI::BarrierDescriptor{prev_ch.stages,
                                 curr_ch.stages,
                                 uint32_t(RHI::DependencyType::NONE),
                                 {},
                                 {},
                                 {}};
      desc.bufferMemoryBarriers.push_back(RHI::BufferMemoryBarrierDescriptor{
          res.second->buffer->buffer.get(), prev_ch.access, curr_ch.access});
      barriers[hentry.passID].emplace_back(desc);
    }
  }
}

auto Graph::generateTextureBarriers() noexcept -> void {
  for (auto& res : textureResources) {
    if (res.second->cosumeHistories.size() == 0) continue;

    TextureResourceVirtualMachine vm(res.second->texture->texture.get(),
                                     res.second->cosumeHistories);
    // deal with all max-possbiel notations
    for (auto& hentry : res.second->cosumeHistories) {
      for (auto& subentry : hentry.entries) {
        if (subentry.mip_end == MaxPossible)
          subentry.mip_end = res.second->texture->texture->mipLevelCount();
      }
    }
    for (auto const& hentry : res.second->cosumeHistories) {
      for (auto const& subentry : hentry.entries) {
        std::vector<RHI::BarrierDescriptor> decses = vm.updateSubresource(
            TextureResourceVirtualMachine::TextureSubresourceRange{
                subentry.level_beg, subentry.level_end, subentry.mip_beg,
                subentry.mip_end},
            TextureResourceVirtualMachine::TextureSubresourceState{
                subentry.stages, subentry.access, subentry.layout});
        for (auto const& desc : decses)
          barriers[hentry.passID].emplace_back(desc);
        if (subentry.type == TextureInfo::ConsumeType::ColorAttachment ||
            subentry.type == TextureInfo::ConsumeType::DepthStencilAttachment) {
          vm.updateSubresource(
              TextureResourceVirtualMachine::TextureSubresourceRange{
                  subentry.level_beg, subentry.level_end, subentry.mip_beg,
                  subentry.mip_end},
              TextureResourceVirtualMachine::TextureSubresourceState{
                  subentry.stages, subentry.access,
                  RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL});
        }
      }
    }
  }

  for (auto& barrier : barriers) {
    tryMergeBarriers(barrier.second);
  }
}
}