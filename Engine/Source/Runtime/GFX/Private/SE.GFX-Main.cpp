#include <SE.GFX-Main.hpp>
#include <IO/SE.Core.IO.hpp>
#include <SE.GFX-Script.hpp>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/yaml.h>

namespace SIByL::GFX {
/**
 * GFX Loader is a singleton sub-module of GFX Manager.
 * It manages how to load resources from
 */
auto Buffer::release() noexcept -> void {
  Core::ResourceManager::get()->removeResource<GFX::Buffer>(guid);
  guid = Core::INVALID_GUID;
  buffer = nullptr;
}

auto TagComponent::serialize(void* pemitter, Core::EntityHandle const& handle)
    -> void {
  YAML::Emitter& emitter = *reinterpret_cast<YAML::Emitter*>(pemitter);
  Core::Entity entity(handle);
  TagComponent* tag = entity.getComponent<TagComponent>();
  if (tag != nullptr) {
    emitter << YAML::Key << "TagComponent";
    std::string const& name = tag->name;
    emitter << YAML::Value << name;
  }
}

auto TagComponent::deserialize(void* compAoS, Core::EntityHandle const& handle)
    -> void {
  YAML::NodeAoS& components = *reinterpret_cast<YAML::NodeAoS*>(compAoS);
  Core::Entity entity(handle);
  auto tagComponentAoS = components["TagComponent"];
  if (tagComponentAoS) {
    entity.getComponent<TagComponent>()->name =
        tagComponentAoS.as<std::string>();
  }
}

auto TransformComponent::getTransform() noexcept -> Math::mat4 {
  return Math::mat4::translate(translation) *
         Math::mat4::rotateZ(eulerAngles.z) *
         Math::mat4::rotateY(eulerAngles.y) *
         Math::mat4::rotateX(eulerAngles.x) * Math::mat4::scale(scale);
}

auto TransformComponent::serialize(void* pemitter,
                                   Core::EntityHandle const& handle) -> void {
  YAML::Emitter& emitter = *reinterpret_cast<YAML::Emitter*>(pemitter);
  Core::Entity entity(handle);
  TransformComponent* transform = entity.getComponent<TransformComponent>();
  if (transform != nullptr) {
    emitter << YAML::Key << "Transform";
    emitter << YAML::Value << YAML::BeginMap;
    emitter << YAML::Key << "translation" << YAML::Value
            << transform->translation;
    emitter << YAML::Key << "eulerAngles" << YAML::Value
            << transform->eulerAngles;
    emitter << YAML::Key << "scale" << YAML::Value << transform->scale;
    emitter << YAML::Key << "staticParam" << YAML::Value << transform->static_param;
    emitter << YAML::EndMap;
  }
}

auto TransformComponent::deserialize(void* compAoS,
                                     Core::EntityHandle const& handle) -> void {
  YAML::NodeAoS& components = *reinterpret_cast<YAML::NodeAoS*>(compAoS);
  Core::Entity entity(handle);
  auto transformComponentAoS = components["Transform"];
  if (transformComponentAoS) {
    TransformComponent* transform = entity.getComponent<TransformComponent>();
    Math::vec3 translation =
        transformComponentAoS["translation"].as<Math::vec3>();
    Math::vec3 eulerAngles =
        transformComponentAoS["eulerAngles"].as<Math::vec3>();
    Math::vec3 scale = transformComponentAoS["scale"].as<Math::vec3>();
    transform->translation = translation;
    transform->eulerAngles = eulerAngles;
    transform->scale = scale;
    if (transformComponentAoS["staticParam"]) {
      transform->static_param = transformComponentAoS["staticParam"].as<uint32_t>();
    }
  }
}

auto to_string(LightComponent::LightType type) noexcept
    -> std::string {
  switch (type) {
    case SIByL::GFX::LightComponent::LightType::DIRECTIONAL:
      return "Directional Light";
    case SIByL::GFX::LightComponent::LightType::POINT:
      return "Point Light";
    case SIByL::GFX::LightComponent::LightType::SPOT:
      return "Spot Light";
    case SIByL::GFX::LightComponent::LightType::TRIANGLE:
      return "Triangle Area Light";
    case SIByL::GFX::LightComponent::LightType::RECTANGLE:
      return "Rectangle Area Light";
    case SIByL::GFX::LightComponent::LightType::MESH_PRIMITIVE:
      return "Mesh Primitive Light";
    case SIByL::GFX::LightComponent::LightType::ENVIRONMENT:
      return "Environment Map";
    case SIByL::GFX::LightComponent::LightType::VPL:
      return "Virtual Point Light";
    default:
      return "Unknown Type";
  }
}

auto AnimationComponent::serialize(void* pemitter,
    Core::EntityHandle const& handle) -> void {
  YAML::Emitter& emitter = *reinterpret_cast<YAML::Emitter*>(pemitter);
  Core::Entity entity(handle);
  AnimationComponent* animComp = entity.getComponent<AnimationComponent>();
  if (animComp != nullptr) {
    emitter << YAML::Key << "AnimationComponent";
    emitter << YAML::Value << YAML::BeginMap;
    emitter << YAML::Key << "Name" << YAML::Value << animComp->ani.name;
    emitter << YAML::Key << "SamplerCount" << YAML::Value << animComp->ani.samplers.size();
    emitter << YAML::Key << "ChannelCount" << YAML::Value << animComp->ani.channels.size();
    emitter << YAML::Key << "Start" << YAML::Value << animComp->ani.start;
    emitter << YAML::Key << "End" << YAML::Value << animComp->ani.end;
    // serialize samplers
    emitter << YAML::Key << "Samplers" << YAML::Value << YAML::BeginSeq;
    for (auto& sampler : animComp->ani.samplers) {
      emitter << YAML::BeginMap;
      emitter << YAML::Key << "InterpolationType" << YAML::Value << uint32_t(sampler.interpolation);
      emitter << YAML::Key << "Inputs" << YAML::Value << YAML::BeginSeq;
      for (int i = 0; i < sampler.inputs.size(); i++)
        emitter << sampler.inputs[i];
      emitter << YAML::EndSeq;
      if (sampler.outputsVec3.size() != 0) {
        emitter << YAML::Key << "ValType" << YAML::Value << "Float3";
        emitter << YAML::Key << "Vals" << YAML::Value << YAML::BeginSeq;
        for (int i = 0; i < sampler.outputsVec3.size(); i++)
          emitter << sampler.outputsVec3[i];
        emitter << YAML::EndSeq;
      } else if (sampler.outputsVec4.size() != 0) {
        emitter << YAML::Key << "ValType" << YAML::Value << "Float4";
        emitter << YAML::Key << "Vals" << YAML::Value << YAML::BeginSeq;
        for (int i = 0; i < sampler.outputsVec4.size(); i++)
          emitter << sampler.outputsVec4[i];
        emitter << YAML::EndSeq;
      }
      emitter << YAML::EndMap;
    }
    emitter << YAML::EndSeq;
    // serialize channels
    emitter << YAML::Key << "Channels" << YAML::Value << YAML::BeginSeq;
    for (auto& channel : animComp->ani.channels) {
      emitter << YAML::BeginMap;
      emitter << YAML::Key << "PathType" << YAML::Value << uint32_t(channel.path);
      emitter << YAML::Key << "SamplerIndex" << YAML::Value << channel.samplerIndex;
      emitter << YAML::EndMap;
    }
    emitter << YAML::EndSeq;
    // endup serialization
    emitter << YAML::EndMap;
  }
}

auto AnimationComponent::deserialize(void* compAoS,
    Core::EntityHandle const& handle) -> void {
  YAML::NodeAoS& components = *reinterpret_cast<YAML::NodeAoS*>(compAoS);
  Core::Entity entity(handle);
  auto animationComponentAoS = components["AnimationComponent"];
  if (animationComponentAoS) {
    AnimationComponent* animComp = entity.addComponent<AnimationComponent>();
    animComp->ani.name = animationComponentAoS["Name"].as<std::string>();
    animComp->ani.start = animationComponentAoS["Start"].as<float>();
    animComp->ani.end = animationComponentAoS["End"].as<float>();
    animComp->ani.samplers.resize(animationComponentAoS["SamplerCount"].as<size_t>());
    animComp->ani.channels.resize(animationComponentAoS["ChannelCount"].as<size_t>());
    // load samplers
    auto samplers = animationComponentAoS["Samplers"];
    uint32_t sampler_idx = 0;
    for (auto sampler_node : samplers) {
      AnimationSampler sampler;
      sampler.interpolation = AnimationSampler::InterpolationType(
          sampler_node["InterpolationType"].as<uint32_t>());
      for (auto input : sampler_node["Inputs"]) {
        sampler.inputs.emplace_back(input.as<float>());
      }
      std::string val_type = sampler_node["ValType"].as<std::string>();
      if (val_type.compare("Float3") == 0)
        for (auto val : sampler_node["Vals"])
          sampler.outputsVec3.emplace_back(val.as<Math::vec3>());
      else if (val_type.compare("Float4") == 0)
        for (auto val : sampler_node["Vals"])
          sampler.outputsVec4.emplace_back(val.as<Math::vec4>());
      animComp->ani.samplers[sampler_idx++] = sampler;
    }
    // load channels
    auto channels = animationComponentAoS["Channels"];
    uint32_t channel_idx = 0;
    for (auto channel_node : channels) {
      AnimationChannel channel;
      channel.samplerIndex = channel_node["SamplerIndex"].as<uint32_t>();
      channel.path = AnimationChannel::PathType(
          channel_node["PathType"].as<uint32_t>());
      animComp->ani.channels[channel_idx++] = channel;
    }
  }
}

/******************************************************************
 * Extension to GFX::Buffer
 *******************************************************************/

  SBTsDescriptor::operator RHI::SBTsDescriptor() const {
  RHI::SBTsDescriptor sbt;
  sbt.rgenSBT.rgenRecord.rayGenShader =
      rgenSBT.rgenRecord.rayGenShader->shaderModule.get();
  for (auto const& entry : missSBT.rmissRecords)
    sbt.missSBT.rmissRecords.push_back(RHI::SBTsDescriptor::MissSBT::MissRecord{
        entry.missShader->shaderModule.get()});
  for (auto const& entry : hitGroupSBT.hitGroupRecords)
    sbt.hitGroupSBT.hitGroupRecords.push_back(
        RHI::SBTsDescriptor::HitGroupSBT::HitGroupRecord{
            entry.closetHitShader == nullptr
                ? nullptr
                : entry.closetHitShader->shaderModule.get(),
            entry.anyHitShader == nullptr
                ? nullptr
                : entry.anyHitShader->shaderModule.get(),
            entry.intersectionShader == nullptr
                ? nullptr
                : entry.intersectionShader->shaderModule.get()});
  for (auto const& entry : callableSBT.callableRecords)
    sbt.callableSBT.callableRecords.push_back(
        RHI::SBTsDescriptor::CallableSBT::CallableRecord{
            entry.callableShader->shaderModule.get()});
  return sbt;
}
#pragma region SCENE_IMPL

auto Scene::createGameObject(GameObjectHandle parent) noexcept
    -> GameObjectHandle {
  Core::Entity entity = Core::EntityManager::get()->createEntity();
  gameObjects.insert({entity.handle, GameObject{parent, entity.handle}});
  if (parent != NULL_GO) gameObjects[parent].children.push_back(entity.handle);
  gameObjects[entity.handle].getEntity().addComponent<TagComponent>(
      "New GameObject");
  gameObjects[entity.handle].getEntity().addComponent<TransformComponent>();
  return GameObjectHandle(entity.handle);
}

auto Scene::removeGameObject(GameObjectHandle handle) noexcept -> void {
  if (gameObjects.find(handle) == gameObjects.end()) return;
  GameObject& go = gameObjects[handle];
  if (go.parent != NULL_GO) {
    // remove the go from its parent's children list
    GameObject& parent = gameObjects[go.parent];
    for (auto iter = parent.children.begin(); iter != parent.children.end();
         ++iter) {
      if (*iter == handle) {
        parent.children.erase(iter);
        break;
      }
    }
  }
  // remove recursively its children
  std::vector<GameObjectHandle> children = go.children;
  for (auto child : children) removeGameObject(child);
  // remove the gameobject
  gameObjects.erase(handle);
  Core::EntityManager::get()->destroyEntity(handle);
}

auto Scene::getGameObject(GameObjectHandle handle) noexcept -> GameObject* {
  if (gameObjects.find(handle) == gameObjects.end())
    return nullptr;
  else
    return &gameObjects[handle];
}

auto Scene::serialize(std::filesystem::path path) noexcept -> void {
  std::unordered_map<GameObjectHandle, uint64_t> mapper;
  uint64_t index = 0;
  mapper[Core::NULL_ENTITY] = Core::NULL_ENTITY;
  for (auto iter = gameObjects.begin(); iter != gameObjects.end(); iter++) {
    mapper[iter->first] = index++;
  }
  YAML::Emitter out;
  out << YAML::BeginMap;
  // output name
  out << YAML::Key << "SceneName" << YAML::Value << name;
  // output nodes
  out << YAML::Key << "SceneNodes" << YAML::Value << YAML::BeginSeq;
  for (auto iter = gameObjects.begin(); iter != gameObjects.end(); iter++) {
    out << YAML::BeginMap;
    // uid
    out << YAML::Key << "uid" << YAML::Value << mapper[iter->first];
    // parent
    out << YAML::Key << "parent" << YAML::Value << mapper[iter->second.parent];
    // children
    if (iter->second.children.size() > 0) {
      out << YAML::Key << "children" << YAML::Value << YAML::BeginSeq;
      for (int i = 0; i < iter->second.children.size(); i++)
        out << mapper[iter->second.children[i]];
      out << YAML::EndSeq;
    }
    // components
    out << YAML::Key << "components" << YAML::Value;
    out << YAML::BeginMap;
    Core::ComponentManager::get()->trySerialize(&out, iter->second.entity);
    out << YAML::EndMap;
    // end
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;
  // output tail
  out << YAML::Key << "SceneEnd" << YAML::Value << "TRUE";
  out << YAML::EndMap;
  Core::Buffer scene_proxy;
  scene_proxy.data = (void*)out.c_str();
  scene_proxy.size = out.size();
  Core::syncWriteFile(path.string().c_str(), scene_proxy);
  scene_proxy.data = nullptr;
}

auto Scene::deserialize(std::filesystem::path path) noexcept -> void {
  release();
  // gameObjects.clear();
  Core::Buffer scene_proxy;
  Core::syncReadFile(path.string().c_str(), scene_proxy);
  YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(scene_proxy.data));
  // check scene name
  if (!data["SceneName"] || !data["SceneNodes"]) {
    Core::LogManager::Error(std::format(
        "GFX :: Scene Name not found when deserializing {0}", path.string()));
    return;
  }
  name = data["SceneName"].as<std::string>();
  std::unordered_map<uint64_t, GameObjectHandle> mapper;
  mapper[Core::NULL_ENTITY] = Core::NULL_ENTITY;
  uint32_t index = 0;
  auto scene_nodes = data["SceneNodes"];
  for (auto node : scene_nodes) {
    uint64_t uid = node["uid"].as<uint64_t>();
    uint64_t parent = node["parent"].as<uint64_t>();
    GameObjectHandle gohandle = createGameObject(Core::NULL_ENTITY);
    GameObject* go = getGameObject(gohandle);
    go->parent = parent;
    auto children = node["children"];
    go->children = std::vector<uint64_t>(children.size());
    uint32_t idx = 0;
    if (children)
      for (auto child : children) go->children[idx++] = child.as<uint64_t>();
    mapper[uid] = gohandle;

    auto components = node["components"];
    Core::ComponentManager::get()->tryDeserialize(&components, gohandle);
  }
  for (auto iter = gameObjects.begin(); iter != gameObjects.end(); iter++) {
    iter->second.parent = mapper[iter->second.parent];
    for (int i = 0; i < iter->second.children.size(); ++i) {
      iter->second.children[i] = mapper[iter->second.children[i]];
    }
  }
}

auto Scene::release() noexcept -> void {
  std::vector<GameObjectHandle> handles;
  for (auto go : gameObjects) {
    handles.emplace_back(go.first);
  }
  for (auto handle : handles) {
    removeGameObject(handle);
  }
  gameObjects.clear();
}

#pragma endregion

#pragma region MESH_IMPL

auto Mesh::surfaceAreaEveryPrimitive(Math::mat4 const& transform) noexcept
    -> std::tuple<float, std::vector<float>> {
  size_t primitiveNum = indexBuffer_host.size / (3 * sizeof(uint32_t));
  uint32_t* indexBuffer = static_cast<uint32_t*>(indexBuffer_host.data);
  float* vertexBuffer = static_cast<float*>(vertexBuffer_host.data);
  size_t vertexStride = vertexBufferLayout.arrayStride / sizeof(float);
  std::vector<float> areas;
  float totalArea = 0.f;
  for (size_t i = 0; i < primitiveNum; ++i) {
    uint32_t i0 = indexBuffer[3 * i + 0];
    uint32_t i1 = indexBuffer[3 * i + 1];
    uint32_t i2 = indexBuffer[3 * i + 2];
    Math::vec3 const& pos0 = *(Math::vec3*)(&(vertexBuffer[i0 * vertexStride]));
    Math::vec3 const& pos1 = *(Math::vec3*)(&(vertexBuffer[i1 * vertexStride]));
    Math::vec3 const& pos2 = *(Math::vec3*)(&(vertexBuffer[i2 * vertexStride]));
    Math::vec3 v0 = Math::vec3(transform * Math::vec4(pos0, 0));
    Math::vec3 v1 = Math::vec3(transform * Math::vec4(pos1, 0));
    Math::vec3 v2 = Math::vec3(transform * Math::vec4(pos2, 0));
    Math::vec3 const e1 = v1 - v0;
    Math::vec3 const e2 = v2 - v0;
    float area = Math::length(Math::cross(e1, e2)) / 2;
    areas.push_back(area);
    totalArea += area;
  }
  return std::make_tuple(totalArea, areas);
}

auto Mesh::serialize() noexcept -> void {
  if (ORID == Core::INVALID_ORID) {
    ORID = Core::requestORID();
  }
  std::filesystem::path metadata_path =
      "./bin/" + std::to_string(ORID) + ".meta";
  std::filesystem::path bindata_path = "./bin/" + std::to_string(ORID) + ".bin";
  // handle metadata
  {
    YAML::Emitter out;
    out << YAML::BeginMap;
    // output type
    out << YAML::Key << "ResourceType" << YAML::Value << "Mesh";
    out << YAML::Key << "Name" << YAML::Value << name;
    out << YAML::Key << "ORID" << YAML::Value << ORID;
    // output VertexBufferLayout
    out << YAML::Key << "VertexBufferLayout" << YAML::Value;
    out << YAML::BeginMap;
    {
      out << YAML::Key << "ArrayStride" << YAML::Value
          << vertexBufferLayout.arrayStride;
      out << YAML::Key << "VertexStepMode" << YAML::Value
          << (uint32_t)vertexBufferLayout.stepMode;
      out << YAML::Key << "VertexAttributes" << YAML::Value << YAML::BeginSeq;
      for (int i = 0; i < vertexBufferLayout.attributes.size(); i++) {
        out << YAML::BeginMap;
        out << YAML::Key << "VertexFormat" << YAML::Value
            << (uint32_t)vertexBufferLayout.attributes[i].format;
        out << YAML::Key << "Offset" << YAML::Value
            << vertexBufferLayout.attributes[i].offset;
        out << YAML::Key << "Location" << YAML::Value
            << vertexBufferLayout.attributes[i].shaderLocation;
        out << YAML::EndMap;
      }
      out << YAML::EndSeq;
    }
    out << YAML::EndMap;
    // output PrimitiveState
    out << YAML::Key << "PrimitiveState" << YAML::Value;
    out << YAML::BeginMap;
    {
      out << YAML::Key << "PrimitiveTopology" << YAML::Value
          << (uint32_t)primitiveState.topology;
      out << YAML::Key << "IndexFormat" << YAML::Value
          << (uint32_t)primitiveState.stripIndexFormat;
      out << YAML::Key << "FrontFace" << YAML::Value
          << (uint32_t)primitiveState.frontFace;
      out << YAML::Key << "CullMode" << YAML::Value
          << (uint32_t)primitiveState.cullMode;
      out << YAML::Key << "UnclippedDepth" << YAML::Value
          << primitiveState.unclippedDepth;
    }
    out << YAML::EndMap;

    out << YAML::Key << "VertexBufferSize" << YAML::Value
        << vertexBufferInfo.size;
    out << YAML::Key << "IndexBufferSize" << YAML::Value
        << indexBufferInfo.size;
    out << YAML::Key << "PosOnlyBufferSize" << YAML::Value
        << positionBufferInfo.size;
    out << YAML::Key << "UV2BufferSize" << YAML::Value
        << uv2Buffer_host.size;
    // output submeshes
    out << YAML::Key << "Submeshes" << YAML::Value << YAML::BeginSeq;
    for (int i = 0; i < submeshes.size(); i++) {
      out << YAML::BeginMap;
      out << YAML::Key << "BaseVertex" << YAML::Value
          << submeshes[i].baseVertex;
      out << YAML::Key << "Offset" << YAML::Value << submeshes[i].offset;
      out << YAML::Key << "Size" << YAML::Value << submeshes[i].size;
      out << YAML::Key << "MatID" << YAML::Value << submeshes[i].matID;
      out << YAML::EndMap;
    }
    out << YAML::EndSeq;
    // output tail
    out << YAML::Key << "End" << YAML::Value << "TRUE";
    out << YAML::EndMap;
    Core::Buffer scene_proxy;
    scene_proxy.data = (void*)out.c_str();
    scene_proxy.size = out.size();
    Core::syncWriteFile(metadata_path.string().c_str(), scene_proxy);
    scene_proxy.data = nullptr;
  }
  // handle binary data
  int vbsize = vertexBufferInfo.size;
  int ibsize = indexBufferInfo.size;
  int pbsize = positionBufferInfo.size;
  int uv2size = uv2Buffer_host.size;
  Core::Buffer mergedBuffer(vbsize + ibsize + pbsize + uv2size);
  if (vertexBufferInfo.onHost) {
    memcpy(mergedBuffer.data, vertexBuffer_host.data, vbsize);
  } else if (vertexBufferInfo.onDevice) {
    vertexBuffer_device->getDevice()->waitIdle();
    vertexBuffer_device->getDevice()->readbackDeviceLocalBuffer(
        vertexBuffer_device.get(), mergedBuffer.data, vbsize);
  }
  if (indexBufferInfo.onHost) {
    memcpy(&(((char*)(mergedBuffer.data))[vbsize]), indexBuffer_host.data,
           ibsize);
  } else if (indexBufferInfo.onDevice) {
    indexBuffer_device->getDevice()->waitIdle();
    indexBuffer_device->getDevice()->readbackDeviceLocalBuffer(
        indexBuffer_device.get(), &(((char*)(mergedBuffer.data))[vbsize]),
        ibsize);
  }
  if (positionBufferInfo.onHost) {
    memcpy(&(((char*)(mergedBuffer.data))[vbsize + ibsize]),
           positionBuffer_host.data, pbsize);
  } else if (positionBufferInfo.onDevice) {
    positionBuffer_device->getDevice()->waitIdle();
    positionBuffer_device->getDevice()->readbackDeviceLocalBuffer(
        positionBuffer_device.get(),
        &(((char*)(mergedBuffer.data))[vbsize + ibsize]), pbsize);
  }
  if (uv2Buffer_host.size != 0) {
    memcpy(&(((char*)(mergedBuffer.data))[vbsize + ibsize + pbsize]),
           uv2Buffer_host.data, uv2size);
  }
  Core::syncWriteFile(bindata_path.string().c_str(), mergedBuffer);
}

auto Mesh::deserialize(RHI::Device* device, Core::ORID orid) noexcept
    -> void {
  ORID = orid;
  std::filesystem::path metadata_path =
      "./bin/" + std::to_string(ORID) + ".meta";
  std::filesystem::path bindata_path = "./bin/" + std::to_string(ORID) + ".bin";

  // gameObjects.clear();
  Core::Buffer metadata;
  Core::syncReadFile(metadata_path.string().c_str(), metadata);
  YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(metadata.data));
  // check scene name
  if (data["ResourceType"].as<std::string>() != "Mesh") {
    Core::LogManager::Error(std::format(
        "GFX :: Mesh resource not found when deserializing, ORID: {0}",
        std::to_string(orid)));
    return;
  }
  name = data["Name"].as<std::string>();
  auto vbbl_node = data["VertexBufferLayout"];
  vertexBufferLayout.arrayStride = vbbl_node["ArrayStride"].as<size_t>();
  vertexBufferLayout.stepMode =
      (RHI::VertexStepMode)vbbl_node["ArrayStride"].as<uint32_t>();
  auto attribute_nodes = vbbl_node["VertexAttributes"];
  for (auto node : attribute_nodes) {
    RHI::VertexAttribute attribute;
    attribute.format = (RHI::VertexFormat)node["VertexFormat"].as<uint32_t>();
    attribute.offset = node["Offset"].as<size_t>();
    attribute.shaderLocation = node["Location"].as<uint32_t>();
    vertexBufferLayout.attributes.push_back(attribute);
  }
  auto ps_node = data["PrimitiveState"];
  primitiveState.topology =
      (RHI::PrimitiveTopology)ps_node["PrimitiveTopology"].as<uint32_t>();
  primitiveState.stripIndexFormat =
      (RHI::IndexFormat)ps_node["IndexFormat"].as<uint32_t>();
  primitiveState.frontFace =
      (RHI::FrontFace)ps_node["FrontFace"].as<uint32_t>();
  primitiveState.cullMode = (RHI::CullMode)ps_node["CullMode"].as<uint32_t>();
  primitiveState.unclippedDepth = ps_node["UnclippedDepth"].as<bool>();
  // load buffers
  size_t vb_size, ib_size, pb_size, uv2_size;
  vb_size = data["VertexBufferSize"].as<size_t>();
  ib_size = data["IndexBufferSize"].as<size_t>();
  pb_size = data["PosOnlyBufferSize"].as<size_t>();
  if (data["UV2BufferSize"]) {
    uv2_size = data["UV2BufferSize"].as<size_t>();
  } else { uv2_size = 0; }

  vertexBufferInfo.size = vb_size;
  indexBufferInfo.size = ib_size;
  positionBufferInfo.size = pb_size;
  // load submeshes
  auto submeshes_node = data["Submeshes"];
  for (auto node : submeshes_node) {
    Submesh submesh;
    submesh.baseVertex = node["BaseVertex"].as<uint32_t>();
    submesh.offset = node["Offset"].as<uint32_t>();
    submesh.size = node["Size"].as<uint32_t>();
    submesh.matID = node["MatID"].as<uint32_t>();
    submeshes.push_back(submesh);
  }
  Core::Buffer bindata;
  Core::syncReadFile(bindata_path.string().c_str(), bindata);

  MeshLoaderConfig meshConfig = GFXConfig::globalConfig->meshLoaderConfig;
  if (meshConfig.residentOnDevice) {
    vertexBufferInfo.onDevice = true;
    indexBufferInfo.onDevice = true;
    positionBufferInfo.onDevice = true;
    vertexBuffer_device = device->createDeviceLocalBuffer(
        (void*)bindata.data, vb_size,
        (uint32_t)RHI::BufferUsage::VERTEX |
            (uint32_t)RHI::BufferUsage::STORAGE);
    indexBuffer_device = device->createDeviceLocalBuffer(
        (void*)&(((uint16_t*)(&(((char*)(bindata.data))[vb_size])))[0]),
        ib_size,
        (uint32_t)RHI::BufferUsage::INDEX |
            (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
            (uint32_t)
                RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY |
            (uint32_t)RHI::BufferUsage::STORAGE);
    if (pb_size != 0) {
      positionBuffer_device = device->createDeviceLocalBuffer(
          (void*)&(
              ((uint16_t*)(&(((char*)(bindata.data))[vb_size + ib_size])))[0]),
          pb_size,
          (uint32_t)RHI::BufferUsage::INDEX |
              (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
              (uint32_t)RHI::BufferUsage::
                  ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY |
              (uint32_t)RHI::BufferUsage::STORAGE);
    }
  } else if (meshConfig.residentOnHost) {
    vertexBufferInfo.onHost = true;
    indexBufferInfo.onHost = true;
    positionBufferInfo.onHost = true;
    vertexBuffer_host = Core::Buffer(vb_size);
    indexBuffer_host = Core::Buffer(ib_size);
    positionBuffer_host = Core::Buffer(pb_size);
    memcpy(vertexBuffer_host.data, (void*)bindata.data, vb_size);
    memcpy(indexBuffer_host.data,
           (void*)&(((uint16_t*)(&(((char*)(bindata.data))[vb_size])))[0]),
           ib_size);
    memcpy(positionBuffer_host.data,
           (void*)&(
               ((uint16_t*)(&(((char*)(bindata.data))[vb_size + ib_size])))[0]),
           pb_size);
    if (uv2_size > 0) {
      uv2Buffer_host = Core::Buffer(uv2_size);
      memcpy(uv2Buffer_host.data, (void*)&(((uint16_t*)(&(
             ((char*)(bindata.data))[vb_size + ib_size + pb_size])))[0]),
             uv2_size);
    }
  }

  auto const& find_aabb = data["AABB"];
  if (find_aabb.Type() == YAML::NodeType::Undefined) {
    size_t pos_num = positionBuffer_host.size / sizeof(float);
    float* position_buffer = static_cast<float*>(positionBuffer_host.data);
    aabb = Math::bounds3{};
    for (size_t i = 0; i < pos_num; i += 3) {
      Math::vec3 pos = {position_buffer[i + 0], position_buffer[i + 1],
                        position_buffer[i + 2]};
      aabb = Math::unionPoint<float>(aabb, pos);
    }
  }
}

#pragma endregion

#pragma region TEXTURE_IMPL

auto Texture::serialize() noexcept -> void {
  // only serialize if has orid
  if (orid != Core::INVALID_ORID && resourcePath.has_value()) {
    std::filesystem::path metadata_path =
        "./bin/" + std::to_string(orid) + ".meta";
    // handle metadata
    {
      YAML::Emitter out;
      out << YAML::BeginMap;
      // output type
      out << YAML::Key << "ResourceType" << YAML::Value << "Texture";
      out << YAML::Key << "Name" << YAML::Value << getName();
      out << YAML::Key << "ORID" << YAML::Value << orid;
      out << YAML::Key << "path" << YAML::Value << resourcePath.value();
      out << YAML::Key << "End" << YAML::Value << "TRUE";
      out << YAML::EndMap;
      Core::Buffer tex_proxy;
      tex_proxy.data = (void*)out.c_str();
      tex_proxy.size = out.size();
      Core::syncWriteFile(metadata_path.string().c_str(), tex_proxy);
      tex_proxy.data = nullptr;
    }
  }
  // only serialize if has orid
  if (orid != Core::INVALID_ORID && resourcePathArray.has_value() &&
      resourcePathArray.value().size() > 1) {
    std::filesystem::path metadata_path =
        "./bin/" + std::to_string(orid) + ".meta";
    // handle metadata
    {
      YAML::Emitter out;
      out << YAML::BeginMap;
      // output type
      out << YAML::Key << "ResourceType" << YAML::Value << "Texture";
      out << YAML::Key << "Name" << YAML::Value << getName();
      out << YAML::Key << "ORID" << YAML::Value << orid;
      out << YAML::Key << "path" << YAML::Value << "USE_PATH_ARRAY";
      out << YAML::Key << "pathArray" << YAML::Value << YAML::BeginSeq;
      for (int i = 0; i < resourcePathArray.value().size(); i++)
        out << resourcePathArray.value()[i];
      out << YAML::EndSeq;
      out << YAML::Key << "End" << YAML::Value << "TRUE";
      out << YAML::EndMap;
      Core::Buffer tex_proxy;
      tex_proxy.data = (void*)out.c_str();
      tex_proxy.size = out.size();
      Core::syncWriteFile(metadata_path.string().c_str(), tex_proxy);
      tex_proxy.data = nullptr;
    }
  }
}

auto Texture::deserialize(RHI::Device* device, Core::ORID ORID) noexcept
    -> void {
  orid = ORID;
  if (orid == Core::INVALID_ORID) return;
  std::filesystem::path metadata_path =
      "./bin/" + std::to_string(ORID) + ".meta";
  Core::Buffer metadata;
  Core::syncReadFile(metadata_path.string().c_str(), metadata);
  YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(metadata.data));
  // check scene name
  if (data["ResourceType"].as<std::string>() != "Texture") {
    Core::LogManager::Error(std::format(
        "GFX :: Texture resource not found when deserializing, ORID: {0}",
        std::to_string(orid)));
    return;
  }
  name = data["Name"].as<std::string>();
  resourcePath = data["path"].as<std::string>();
  if (resourcePath == "USE_PATH_ARRAY") {
    std::vector<std::string> pathArray = {};
    for (auto child : data["pathArray"])
      pathArray.push_back(child.as<std::string>());
    resourcePathArray = pathArray;
  }
}

auto findDimension(RHI::TextureDimension dim, uint32_t arraySize) noexcept
    -> RHI::TextureViewDimension {
  RHI::TextureViewDimension dimension;
  switch (dim) {
    case SIByL::RHI::TextureDimension::TEX1D:
      dimension = (arraySize > 1) ? RHI::TextureViewDimension::TEX1D_ARRAY
                                  : RHI::TextureViewDimension::TEX1D;
      break;
    case SIByL::RHI::TextureDimension::TEX2D:
      dimension = (arraySize > 1) ? RHI::TextureViewDimension::TEX2D_ARRAY
                                  : RHI::TextureViewDimension::TEX2D;
      break;
    case SIByL::RHI::TextureDimension::TEX3D:
      dimension = (arraySize > 1) ? RHI::TextureViewDimension::TEX3D_ARRAY
                                  : RHI::TextureViewDimension::TEX3D;
      break;
    default:
      break;
  }
  return dimension;
}

auto Texture::getUAV(uint32_t mipLevel, uint32_t firstArraySlice,
                     uint32_t arraySize) noexcept -> RHI::TextureView* {
  ViewIndex idx = {RHI::TextureViewType::UAV, mipLevel, 0, firstArraySlice,
                   arraySize};
  RHI::TextureViewDimension dimension =
      findDimension(texture->dimension(), arraySize);
  auto find = viewPool.find(idx);
  if (find == viewPool.end()) {
    viewPool[idx] = texture->createView(RHI::TextureViewDescriptor{
        texture->format(), dimension, (uint32_t)RHI::TextureAspect::COLOR_BIT,
        mipLevel, 1, firstArraySlice, arraySize});
    find = viewPool.find(idx);
  }
  return find->second.get();
}

auto Texture::getRTV(uint32_t mipLevel, uint32_t firstArraySlice,
                     uint32_t arraySize) noexcept -> RHI::TextureView* {
  ViewIndex idx = {RHI::TextureViewType::RTV, mipLevel, 0, firstArraySlice,
                   arraySize};
  RHI::TextureViewDimension dimension =
      findDimension(texture->dimension(), arraySize);
  auto find = viewPool.find(idx);
  if (find == viewPool.end()) {
    viewPool[idx] = texture->createView(RHI::TextureViewDescriptor{
        texture->format(), dimension, (uint32_t)RHI::TextureAspect::COLOR_BIT,
        mipLevel, 1, firstArraySlice, arraySize});
    find = viewPool.find(idx);
  }
  return find->second.get();
}

auto Texture::getDSV(uint32_t mipLevel, uint32_t firstArraySlice,
                     uint32_t arraySize) noexcept -> RHI::TextureView* {
  ViewIndex idx = {RHI::TextureViewType::DSV, mipLevel, 0, firstArraySlice,
                   arraySize};
  RHI::TextureViewDimension dimension =
      findDimension(texture->dimension(), arraySize);
  auto find = viewPool.find(idx);
  if (find == viewPool.end()) {
    viewPool[idx] = texture->createView(RHI::TextureViewDescriptor{
        texture->format(), dimension, (uint32_t)RHI::TextureAspect::DEPTH_BIT,
        mipLevel, 1, firstArraySlice, arraySize});
    find = viewPool.find(idx);
  }
  return find->second.get();
}

auto Texture::getSRV(uint32_t mostDetailedMip, uint32_t mipCount,
                     uint32_t firstArraySlice, uint32_t arraySize) noexcept
    -> RHI::TextureView* {
  ViewIndex idx = {RHI::TextureViewType::RTV, mostDetailedMip, mipCount,
                   firstArraySlice, arraySize};
  RHI::TextureViewDimension dimension =
      findDimension(texture->dimension(), arraySize);
  uint32_t aspect = (uint32_t)RHI::TextureAspect::COLOR_BIT;
  if (RHI::hasDepthBit(texture->format()))
    aspect = (uint32_t)RHI::TextureAspect::DEPTH_BIT;
  if (RHI::hasStencilBit(texture->format()))
    aspect |= (uint32_t)RHI::TextureAspect::STENCIL_BIT;

  auto find = viewPool.find(idx);
  if (find == viewPool.end()) {
    viewPool[idx] = texture->createView(RHI::TextureViewDescriptor{
        texture->format(), dimension, aspect, mostDetailedMip, mipCount,
        firstArraySlice, arraySize});
    find = viewPool.find(idx);
  }
  return find->second.get();
}

#pragma endregion

#pragma region MATERIAL_TEMPLATE_IMPL

auto MaterialTemplate::addConstantData(std::string const& name,
                                              RHI::DataFormat format) noexcept
    -> MaterialTemplate& {
  constDataEntries[name] = format;
  return *this;
}

auto MaterialTemplate::addTexture(std::string const& name) noexcept
    -> MaterialTemplate& {
  textureEntries.push_back(name);
  return *this;
}

#pragma endregion

#pragma region MATERIAL_IMPL

auto Material::addConstantData(std::string const& name,
                                      RHI::DataFormat format) noexcept
    -> Material& {
  return *this;
}

auto Material::addTexture(std::string const& name,
                                 TextureEntry const& entry) noexcept
    -> Material& {
  textures[name] = entry;
  return *this;
}

auto Material::registerFromTemplate(
    MaterialTemplate const& mat_template) noexcept -> void {
  // add datas

  // add textures
  for (auto const& tex : mat_template.textureEntries)
    addTexture(tex, {Core::INVALID_GUID, 0});
}

auto Material::serialize() noexcept -> void {
  if (ORID == Core::INVALID_ORID) {
    ORID = Core::requestORID();
  }
  if (path == "") {
    path = "./content/materials/" + std::to_string(ORID) + ".mat";
  }
  std::filesystem::path matdata_path = path;
  std::filesystem::path metadata_path =
      "./bin/" + std::to_string(ORID) + ".meta";
  // handle metadata
  {// handle metadata
   {YAML::Emitter out;
  out << YAML::BeginMap;
  // output type
  out << YAML::Key << "ResourceType" << YAML::Value << "MaterialPtr";
  out << YAML::Key << "Name" << YAML::Value << getName();
  out << YAML::Key << "ORID" << YAML::Value << ORID;
  out << YAML::Key << "path" << YAML::Value << path;
  out << YAML::Key << "End" << YAML::Value << "TRUE";
  out << YAML::EndMap;
  Core::Buffer mat_proxy;
  mat_proxy.data = (void*)out.c_str();
  mat_proxy.size = out.size();
  Core::syncWriteFile(metadata_path.string().c_str(), mat_proxy);
  mat_proxy.data = nullptr;
}
}  // namespace SIByL::GFX
// handle matdata
{
  YAML::Emitter out;
  out << YAML::BeginMap;
  // output type
  out << YAML::Key << "ResourceType" << YAML::Value << "Material";
  out << YAML::Key << "Name" << YAML::Value << name;
  out << YAML::Key << "BxDF" << YAML::Value << BxDF;
  // output texture
  out << YAML::Key << "Textures" << YAML::Value;
  out << YAML::BeginSeq;
  {
    for (auto& [name, entry] : textures) {
      GFX::Texture* texture =
          Core::ResourceManager::get()->getResource<GFX::Texture>(entry.guid);
      out << YAML::BeginMap;
      out << YAML::Key << "Name" << YAML::Value << name;
      out << YAML::Key << "ORID" << YAML::Value
          << ((texture != nullptr) ? texture->orid : 0);
      out << YAML::Key << "flags" << YAML::Value << entry.flags;
      out << YAML::EndMap;
    }
  }
  out << YAML::EndSeq;
  // output data
  out << YAML::Key << "BaseOrDiffuseColor" << YAML::Value << baseOrDiffuseColor;
  out << YAML::Key << "SpecularColor" << YAML::Value << specularColor;
  out << YAML::Key << "EmissiveColor" << YAML::Value << emissiveColor;
  out << YAML::Key << "Roughness" << YAML::Value << roughness;
  out << YAML::Key << "Metalness" << YAML::Value << metalness;
  out << YAML::Key << "Eta" << YAML::Value << eta;
  // output emissive
  out << YAML::Key << "Emissive" << YAML::Value << isEmissive;
  // output emissive
  out << YAML::Key << "AlphaState" << YAML::Value << uint32_t(alphaState);
  out << YAML::Key << "AlphaThreshold" << YAML::Value << alphaThreshold;
  // output tail
  out << YAML::Key << "End" << YAML::Value << "TRUE";
  out << YAML::EndMap;
  Core::Buffer mat_proxy;
  mat_proxy.data = (void*)out.c_str();
  mat_proxy.size = out.size();
  Core::syncWriteFile(matdata_path.string().c_str(), mat_proxy);
  mat_proxy.data = nullptr;
}
}

auto Material::deserialize(RHI::Device* device, Core::ORID orid) noexcept
    -> void {
  ORID = orid;
  std::filesystem::path metadata_path =
      "./bin/" + std::to_string(orid) + ".meta";
  {
    Core::Buffer metadata;
    Core::syncReadFile(metadata_path.string().c_str(), metadata);
    YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(metadata.data));
    // check scene name
    if (data["ResourceType"].as<std::string>() != "MaterialPtr") {
      Core::LogManager::Error(std::format(
          "GFX :: MaterialPtr resource not found when deserializing, ORID: {0}",
          std::to_string(orid)));
      return;
    }
    name = data["Name"].as<std::string>();
    path = data["path"].as<std::string>();
  }
  loadPath();
}

auto Material::loadPath() noexcept -> void {
  // load data
  Core::Buffer matdata;
  Core::syncReadFile(path.c_str(), matdata);
  YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(matdata.data));
  // check resource type
  if (data["ResourceType"].as<std::string>() != "Material") {
    Core::LogManager::Error(std::format(
        "GFX :: Material resource not found when deserializing, path: {0}",
        path));
    return;
  }
  BxDF = data["BxDF"].as<uint32_t>();
  auto texture_nodes = data["Textures"];
  for (auto node : texture_nodes) {
    std::string tex_name = node["Name"].as<std::string>();
    Core::ORID orid = node["ORID"].as<Core::ORID>();
    if (orid == 0) continue;
    uint32_t flags = node["flags"].as<uint32_t>();
    if (flags & uint32_t(TexFlag::VideoClip)) {
      textures[tex_name] = {
          *reinterpret_cast<Core::GUID*>(GFXManager::get()
                                             ->getExt<Extension>(Ext::VideoClip)
                                             ->foo(0, &orid)),
          flags};
    } else {
      textures[tex_name] = {
          GFXManager::get()->requestOfflineTextureResource(orid), flags};
    }
  }
  isEmissive = data["Emissive"].as<bool>();
  if (data["AlphaThreshold"]) {
    alphaThreshold = data["AlphaThreshold"].as<float>();
  }
  else {
    alphaThreshold = 1.f;
  }
  if (data["AlphaState"]) {
    alphaState = AlphaState(data["AlphaState"].as<uint32_t>());
  } else {
    alphaState = AlphaState::Opaque;
  }

  if (data["BaseOrDiffuseColor"]) {
    baseOrDiffuseColor = data["BaseOrDiffuseColor"].as<Math::vec3>();
  }
  if (data["SpecularColor"]) {
    specularColor = data["SpecularColor"].as<Math::vec3>();
  }
  if (data["Eta"]) {
    eta = data["Eta"].as<float>();
  }
  if (data["EmissiveColor"]) {
    emissiveColor = data["EmissiveColor"].as<Math::vec3>();
  }
  if (data["Roughness"]) {
    roughness = data["Roughness"].as<float>();
  }
  if (data["Metalness"]) {
    metalness = data["Metalness"].as<float>();
  }
}

#pragma endregion

#pragma region CAMERA_COMPONENT_IMPL

auto CameraComponent::getViewMat() noexcept -> Math::mat4 { return view; }

auto CameraComponent::getProjectionMat() const noexcept->Math::mat4 {
  Math::mat4 projection;
  if (projectType == ProjectType::PERSPECTIVE) {
    projection = Math::perspective(fovy, aspect, near, far).m;
  } else if (projectType == ProjectType::ORTHOGONAL) {
    projection = Math::ortho(-aspect * bottom_top, aspect * bottom_top,
                             -bottom_top, bottom_top, near, far).m;
  }
  return projection;
}

auto CameraComponent::serialize(void* pemitter,
                                Core::EntityHandle const& handle) -> void {
  YAML::Emitter& emitter = *reinterpret_cast<YAML::Emitter*>(pemitter);
  Core::Entity entity(handle);
  CameraComponent* camera = entity.getComponent<CameraComponent>();
  if (camera != nullptr) {
    emitter << YAML::Key << "CameraComponent";
    emitter << YAML::Value << YAML::BeginMap;
    emitter << YAML::Key << "fovy" << YAML::Value << camera->fovy;
    emitter << YAML::Key << "aspect" << YAML::Value << camera->aspect;
    emitter << YAML::Key << "near" << YAML::Value << camera->near;
    emitter << YAML::Key << "far" << YAML::Value << camera->far;
    emitter << YAML::Key << "ProjectType" << YAML::Value
            << (uint32_t)camera->projectType;
    emitter << YAML::Key << "IsPrimary" << YAML::Value
            << camera->isPrimaryCamera;
    emitter << YAML::EndMap;
  }
}

auto CameraComponent::deserialize(void* compAoS,
                                  Core::EntityHandle const& handle) -> void {
  YAML::NodeAoS& components = *reinterpret_cast<YAML::NodeAoS*>(compAoS);
  Core::Entity entity(handle);
  auto cameraComponentAoS = components["CameraComponent"];
  if (cameraComponentAoS) {
    CameraComponent* camRef = entity.addComponent<CameraComponent>();
    camRef->fovy = cameraComponentAoS["fovy"].as<float>();
    camRef->aspect = cameraComponentAoS["aspect"].as<float>();
    camRef->near = cameraComponentAoS["near"].as<float>();
    camRef->far = cameraComponentAoS["far"].as<float>();
    camRef->projectType =
        (ProjectType)cameraComponentAoS["ProjectType"].as<uint32_t>();
    camRef->isPrimaryCamera = cameraComponentAoS["IsPrimary"].as<bool>();
  }
}

#pragma endregion

#pragma region MESH_REFERENCE_COMPONENT_IMPL

auto MeshReference::serialize(void* pemitter, Core::EntityHandle const& handle)
    -> void {
  YAML::Emitter& emitter = *reinterpret_cast<YAML::Emitter*>(pemitter);
  Core::Entity entity(handle);
  MeshReference* meshRef = entity.getComponent<MeshReference>();
  if (meshRef != nullptr) {
    emitter << YAML::Key << "MeshReference";
    emitter << YAML::Value << YAML::BeginMap;
    emitter << YAML::Key << "ORID" << YAML::Value
            << ((meshRef->mesh == nullptr) ? Core::INVALID_ORID
                                           : meshRef->mesh->ORID);
    emitter << YAML::Key << "CPF" << YAML::Value
            << meshRef->customPrimitiveFlag;
    emitter << YAML::EndMap;
  }
}

auto MeshReference::deserialize(void* compAoS, Core::EntityHandle const& handle)
    -> void {
  YAML::NodeAoS& components = *reinterpret_cast<YAML::NodeAoS*>(compAoS);
  Core::Entity entity(handle);
  auto meshRefComponentAoS = components["MeshReference"];
  if (meshRefComponentAoS) {
    MeshReference* meshRef = entity.addComponent<MeshReference>();
    Core::ORID orid = meshRefComponentAoS["ORID"].as<uint64_t>();
    Core::GUID guid =
        Core::ResourceManager::get()->requestRuntimeGUID<GFX::Mesh>();
    if (orid == Core::INVALID_ORID) {
      meshRef->mesh = nullptr;
    } else {
      GFX::Mesh mesh;
      Core::ResourceManager::get()->addResource(guid, std::move(mesh));
      Core::ResourceManager::get()->getResource<GFX::Mesh>(guid)->deserialize(
          RHI::RHILayer::get()->getDevice(), orid);
      meshRef->mesh =
          Core::ResourceManager::get()->getResource<GFX::Mesh>(guid);
    }
    meshRef->customPrimitiveFlag = meshRefComponentAoS["CPF"].as<size_t>();
  }
}

#pragma endregion

#pragma region MESH_FILTER_COMPONENT_IMPL

auto MeshRenderer::serialize(void* pemitter, Core::EntityHandle const& handle)
    -> void {
  YAML::Emitter& emitter = *reinterpret_cast<YAML::Emitter*>(pemitter);
  Core::Entity entity(handle);
  MeshRenderer* renderer = entity.getComponent<MeshRenderer>();
  if (renderer != nullptr) {
    emitter << YAML::Key << "MeshRendererComponent";
    emitter << YAML::Value << YAML::BeginSeq;
    for (auto& material : renderer->materials) {
      if (material == nullptr) continue;
      emitter << YAML::BeginMap;
      emitter << YAML::Key << "ORID" << YAML::Value << material->ORID;
      emitter << YAML::EndMap;
    }
    emitter << YAML::EndSeq;
  }
}

auto MeshRenderer::deserialize(void* compAoS, Core::EntityHandle const& handle)
    -> void {
  YAML::NodeAoS& components = *reinterpret_cast<YAML::NodeAoS*>(compAoS);
  Core::Entity entity(handle);
  auto meshRendererComponentAoS = components["MeshRendererComponent"];
  if (meshRendererComponentAoS) {
    MeshRenderer* meshRenderer = entity.addComponent<MeshRenderer>();
    for (auto node : meshRendererComponentAoS) {
      Core::GUID guid = GFX::GFXManager::get()->requestOfflineMaterialResource(
          node["ORID"].as<Core::ORID>());
      GFX::Material* material =
          Core::ResourceManager::get()->getResource<GFX::Material>(guid);
      meshRenderer->materials.push_back(material);
    }
  }
}

#pragma endregion

#pragma region LIGHT_COMPONENT_IMPL

auto LightComponent::serialize(void* pemitter, Core::EntityHandle const& handle)
    -> void {
  YAML::Emitter& emitter = *reinterpret_cast<YAML::Emitter*>(pemitter);
  Core::Entity entity(handle);
  LightComponent* lightComponent = entity.getComponent<LightComponent>();
  if (lightComponent != nullptr) {
    emitter << YAML::Key << "LightComponent";
    emitter << YAML::Value << YAML::BeginMap;
    emitter << YAML::Key << "Type" << YAML::Value
            << uint32_t(lightComponent->type);
    emitter << YAML::Key << "Intensity" << YAML::Value
            << lightComponent->intensity;
    if (lightComponent->texture) {
      emitter << YAML::Key << "Texture" << YAML::Value
              << lightComponent->texture->orid;
    } else
      emitter << YAML::Key << "Texture" << YAML::Value << Core::INVALID_ORID;

    emitter << YAML::Key << "Pack0" << YAML::Value
            << lightComponent->packed_data_0;
    emitter << YAML::Key << "Pack1" << YAML::Value
            << lightComponent->packed_data_1;


    emitter << YAML::EndMap;
  }
}

auto LightComponent::deserialize(void* compAoS,
                                 Core::EntityHandle const& handle) -> void {
  YAML::NodeAoS& components = *reinterpret_cast<YAML::NodeAoS*>(compAoS);
  Core::Entity entity(handle);
  auto lightComponentAoS = components["LightComponent"];
  if (lightComponentAoS) {
    LightComponent* lightComponent = entity.addComponent<LightComponent>();
    lightComponent->type = LightType(lightComponentAoS["Type"].as<uint32_t>());
    lightComponent->intensity = lightComponentAoS["Intensity"].as<Math::vec3>();
    Core::ORID orid = lightComponentAoS["Texture"].as<uint64_t>();
    if (orid != Core::INVALID_ORID) {
      Core::GUID guid =
          GFX::GFXManager::get()->requestOfflineTextureResource(orid);
      lightComponent->texture =
          Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
    } else {
      lightComponent->texture = nullptr;
    }
    if (lightComponentAoS["Pack0"]) {
      lightComponent->packed_data_0 =
          lightComponentAoS["Pack0"].as<Math::vec4>();
    }
    if (lightComponentAoS["Pack1"]) {
      lightComponent->packed_data_1 =
          lightComponentAoS["Pack1"].as<Math::vec4>();
    }
  }
}

#pragma endregion

#pragma region GFX_MANAGER_IMPL

GFXManager* GFXManager::singleton = nullptr;

auto GFXManager::startUp() noexcept -> void {
  // set singleton
  singleton = this;
  // register component types
  Core::ComponentManager::get()->registerComponent<GFX::TagComponent>();
  Core::ComponentManager::get()->registerComponent<GFX::TransformComponent>();
  Core::ComponentManager::get()->registerComponent<GFX::MeshReference>();
  Core::ComponentManager::get()->registerComponent<GFX::MeshRenderer>();
  Core::ComponentManager::get()->registerComponent<GFX::CameraComponent>();
  Core::ComponentManager::get()->registerComponent<GFX::LightComponent>();
  Core::ComponentManager::get()->registerComponent<GFX::AnimationComponent>();
  Core::ComponentManager::get()->registerComponent<NativeScriptComponent>();
  // register resource types
  Core::ResourceManager::get()->registerResource<GFX::Buffer>();
  Core::ResourceManager::get()->registerResource<GFX::Mesh>();
  Core::ResourceManager::get()->registerResource<GFX::Texture>();
  Core::ResourceManager::get()->registerResource<GFX::Sampler>();
  Core::ResourceManager::get()->registerResource<GFX::ShaderModule>();
  Core::ResourceManager::get()->registerResource<GFX::Material>();
  Core::ResourceManager::get()->registerResource<GFX::Scene>();
  // also startup all extensions
  for (auto& pair : extensions) pair.second->startUp();
  // bind global config
  GFXConfig::globalConfig = &config;
}

auto GFXManager::shutDown() noexcept -> void {
  GFXConfig::globalConfig = nullptr;
}

auto GFXManager::onUpdate() noexcept -> void {
  // update timeline
  if (mTimeline.play) {
    mTimeline.currentSec += mTimeline.timer->deltaTime();
  }
  // update extension
  for (auto& ext : extensions) ext.second->onUpdate();
}

auto GFXManager::registerMaterialTemplate(uint32_t bsdf_id,
                                          std::string const& name) noexcept
    -> MaterialTemplate& {
  auto& mat_temp = material_templates[bsdf_id];
  mat_temp.bsdf_id = bsdf_id;
  mat_temp.materialName = name;
  return mat_temp;
}

auto GFXManager::getMaterialTemplate(uint32_t bsdf_id) noexcept
    -> MaterialTemplate* {
  auto iter = material_templates.find(bsdf_id);
  if (iter == material_templates.end()) return nullptr;
  return &(iter->second);
}

auto GFXManager::registerBufferResource(
    Core::GUID guid, RHI::BufferDescriptor const& desc) noexcept -> void {
  GFX::Buffer bufferResource = {};
  bufferResource.buffer = rhiLayer->getDevice()->createBuffer(desc);
  Core::ResourceManager::get()->addResource(guid, std::move(bufferResource));
}

auto GFXManager::registerBufferResource(Core::GUID guid, void* data,
                                        uint32_t size,
                                        RHI::BufferUsagesFlags usage) noexcept
    -> void {
  GFX::Buffer bufferResource = {};
  if (size > 0) {
      bufferResource.buffer =
          rhiLayer->getDevice()->createDeviceLocalBuffer(data, size, usage);
  } else {
      bufferResource.buffer = nullptr;
  }
  Core::ResourceManager::get()->addResource(guid, std::move(bufferResource));
}

auto GFXManager::registerTextureResource(
    Core::GUID guid, Image::Image<Image::COLOR_R8G8B8A8_UINT>* image) noexcept
    -> void {
  GFX::Texture textureResource = {};
  RHI::BufferDescriptor stagingBufferDescriptor;
  stagingBufferDescriptor.size = image->data.size;
  stagingBufferDescriptor.usage = (uint32_t)RHI::BufferUsage::COPY_SRC;
  stagingBufferDescriptor.memoryProperties =
      (uint32_t)RHI::MemoryProperty::HOST_VISIBLE_BIT |
      (uint32_t)RHI::MemoryProperty::HOST_COHERENT_BIT;
  stagingBufferDescriptor.mappedAtCreation = true;
  std::unique_ptr<RHI::Buffer> stagingBuffer =
      rhiLayer->getDevice()->createBuffer(stagingBufferDescriptor);
  std::future<bool> mapped =
      stagingBuffer->mapAsync(0, 0, stagingBufferDescriptor.size);
  if (mapped.get()) {
    void* mapdata =
        stagingBuffer->getMappedRange(0, stagingBufferDescriptor.size);
    memcpy(mapdata, image->data.data, (size_t)stagingBufferDescriptor.size);
    stagingBuffer->unmap();
  }
  std::unique_ptr<RHI::CommandEncoder> commandEncoder =
      rhiLayer->getDevice()->createCommandEncoder({nullptr});
  // create texture image
  textureResource.texture =
      rhiLayer->getDevice()->createTexture(RHI::TextureDescriptor{
          {(uint32_t)image->width, (uint32_t)image->height, 1},
          1,
          1,
          1,
          RHI::TextureDimension::TEX2D,
          RHI::TextureFormat::RGBA8_UNORM,
          (uint32_t)RHI::TextureUsage::COPY_DST |
              (uint32_t)RHI::TextureUsage::TEXTURE_BINDING,
          {RHI::TextureFormat::RGBA8_UNORM}});

  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::TOP_OF_PIPE_BIT,
      (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {},
      {},
      {RHI::TextureMemoryBarrierDescriptor{
          textureResource.texture.get(),
          RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,
                                     1, 0, 1},
          (uint32_t)RHI::AccessFlagBits::NONE,
          (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
          RHI::TextureLayout::UNDEFINED,
          RHI::TextureLayout::TRANSFER_DST_OPTIMAL}}});

  commandEncoder->copyBufferToTexture(
      {0, 0, 0, stagingBuffer.get()},
      {textureResource.texture.get(),
       0,
       {},
       (uint32_t)RHI::TextureAspect::COLOR_BIT},
      {textureResource.texture->width(), textureResource.texture->height(), 1});

  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
      (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {},
      {},
      {RHI::TextureMemoryBarrierDescriptor{
          textureResource.texture.get(),
          RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,
                                     1, 0, 1},
          (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
          (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
          RHI::TextureLayout::TRANSFER_DST_OPTIMAL,
          RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL}}});

  rhiLayer->getDevice()->getGraphicsQueue()->submit(
      {commandEncoder->finish({})});
  rhiLayer->getDevice()->getGraphicsQueue()->waitIdle();
  textureResource.originalView = textureResource.texture->createView(
      RHI::TextureViewDescriptor{RHI::TextureFormat::RGBA8_UNORM});
  textureResource.guid = guid;
  Core::ResourceManager::get()->addResource(guid, std::move(textureResource));
}

auto GFXManager::registerTextureResource(
    Core::GUID guid, Image::Image<Image::COLOR_R32G32B32A32_FLOAT>* image) noexcept
    -> void {
  GFX::Texture textureResource = {};
  RHI::BufferDescriptor stagingBufferDescriptor;
  stagingBufferDescriptor.size = image->data.size;
  stagingBufferDescriptor.usage = (uint32_t)RHI::BufferUsage::COPY_SRC;
  stagingBufferDescriptor.memoryProperties =
      (uint32_t)RHI::MemoryProperty::HOST_VISIBLE_BIT |
      (uint32_t)RHI::MemoryProperty::HOST_COHERENT_BIT;
  stagingBufferDescriptor.mappedAtCreation = true;
  std::unique_ptr<RHI::Buffer> stagingBuffer =
      rhiLayer->getDevice()->createBuffer(stagingBufferDescriptor);
  std::future<bool> mapped =
      stagingBuffer->mapAsync(0, 0, stagingBufferDescriptor.size);
  if (mapped.get()) {
    void* mapdata =
        stagingBuffer->getMappedRange(0, stagingBufferDescriptor.size);
    memcpy(mapdata, image->data.data, (size_t)stagingBufferDescriptor.size);
    stagingBuffer->unmap();
  }
  std::unique_ptr<RHI::CommandEncoder> commandEncoder =
      rhiLayer->getDevice()->createCommandEncoder({nullptr});
  // create texture image
  textureResource.texture =
      rhiLayer->getDevice()->createTexture(RHI::TextureDescriptor{
          {(uint32_t)image->width, (uint32_t)image->height, 1},
          1,
          1,
          1,
          RHI::TextureDimension::TEX2D,
          RHI::TextureFormat::RGBA32_FLOAT,
          (uint32_t)RHI::TextureUsage::COPY_DST |
          (uint32_t)RHI::TextureUsage::COPY_SRC |
          (uint32_t)RHI::TextureUsage::STORAGE_BINDING |
          (uint32_t)RHI::TextureUsage::TEXTURE_BINDING,
          {RHI::TextureFormat::RGBA32_FLOAT}});

  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::TOP_OF_PIPE_BIT,
      (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {},
      {},
      {RHI::TextureMemoryBarrierDescriptor{
          textureResource.texture.get(),
          RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,
                                     1, 0, 1},
          (uint32_t)RHI::AccessFlagBits::NONE,
          (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
          RHI::TextureLayout::UNDEFINED,
          RHI::TextureLayout::TRANSFER_DST_OPTIMAL}}});

  commandEncoder->copyBufferToTexture(
      {0, 0, 0, stagingBuffer.get()},
      {textureResource.texture.get(),
       0,
       {},
       (uint32_t)RHI::TextureAspect::COLOR_BIT},
      {textureResource.texture->width(), textureResource.texture->height(), 1});

  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
      (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {},
      {},
      {RHI::TextureMemoryBarrierDescriptor{
          textureResource.texture.get(),
          RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,
                                     1, 0, 1},
          (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
          (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
          RHI::TextureLayout::TRANSFER_DST_OPTIMAL,
          RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL}}});

  rhiLayer->getDevice()->getGraphicsQueue()->submit(
      {commandEncoder->finish({})});
  rhiLayer->getDevice()->getGraphicsQueue()->waitIdle();
  textureResource.originalView = textureResource.texture->createView(
      RHI::TextureViewDescriptor{RHI::TextureFormat::RGBA32_FLOAT});
  textureResource.guid = guid;
  Core::ResourceManager::get()->addResource(guid, std::move(textureResource));
}

auto GFXManager::registerTextureResource(Core::GUID guid,
                                         Image::Texture_Host* image) noexcept
    -> void {
  RHI::BufferDescriptor stagingBufferDescriptor;
  stagingBufferDescriptor.size = image->data_size;
  stagingBufferDescriptor.usage = (uint32_t)RHI::BufferUsage::COPY_SRC;
  stagingBufferDescriptor.memoryProperties =
      (uint32_t)RHI::MemoryProperty::HOST_VISIBLE_BIT |
      (uint32_t)RHI::MemoryProperty::HOST_COHERENT_BIT;
  stagingBufferDescriptor.mappedAtCreation = true;
  std::unique_ptr<RHI::Buffer> stagingBuffer =
      rhiLayer->getDevice()->createBuffer(stagingBufferDescriptor);
  std::future<bool> mapped =
      stagingBuffer->mapAsync(0, 0, stagingBufferDescriptor.size);
  if (mapped.get()) {
    void* mapdata =
        stagingBuffer->getMappedRange(0, stagingBufferDescriptor.size);
    memcpy(mapdata, image->getData(), (size_t)stagingBufferDescriptor.size);
    stagingBuffer->unmap();
  }
  std::unique_ptr<RHI::CommandEncoder> commandEncoder =
      rhiLayer->getDevice()->createCommandEncoder({nullptr});
  // create texture image
  GFX::Texture textureResource = {};
  textureResource.texture =
      rhiLayer->getDevice()->createTexture(image->getDescriptor());

  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::TOP_OF_PIPE_BIT,
      (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {},
      {},
      {RHI::TextureMemoryBarrierDescriptor{
          textureResource.texture.get(),
          RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,
                                     image->mip_levels, 0, image->array_layers},
          (uint32_t)RHI::AccessFlagBits::NONE,
          (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
          RHI::TextureLayout::UNDEFINED,
          RHI::TextureLayout::TRANSFER_DST_OPTIMAL}}});

  for (auto const& subresource : image->subResources) {
    commandEncoder->copyBufferToTexture(
        {subresource.offset, 0, 0, stagingBuffer.get()},
        {textureResource.texture.get(),
         subresource.mip,
         {},
         (uint32_t)RHI::TextureAspect::COLOR_BIT},
        {subresource.width, subresource.height, 1});
  }

  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
      (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {},
      {},
      {RHI::TextureMemoryBarrierDescriptor{
          textureResource.texture.get(),
          RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,
                                     image->mip_levels, 0, image->array_layers},
          (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
          (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
          RHI::TextureLayout::TRANSFER_DST_OPTIMAL,
          RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL}}});

  rhiLayer->getDevice()->getGraphicsQueue()->submit(
      {commandEncoder->finish({})});
  rhiLayer->getDevice()->getGraphicsQueue()->waitIdle();
  textureResource.originalView = textureResource.texture->createView(
      RHI::TextureViewDescriptor{image->format});
  textureResource.guid = guid;
  Core::ResourceManager::get()->addResource(guid, std::move(textureResource));
}

auto GFXManager::registerTextureResourceCubemap(
    Core::GUID guid, std::array<char const*, 6> images) noexcept -> void {
  std::array<std::unique_ptr<Image::Image<Image::COLOR_R8G8B8A8_UINT>>, 6>
      imageLoaded;
  std::array<Image::Image<Image::COLOR_R8G8B8A8_UINT>*, 6> imageLoadedArray;
  std::vector<std::string> pathArray;
  for (uint32_t i = 0; i < 6; ++i) {
    std::filesystem::path path(images[i]);
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path relative_path =
        std::filesystem::relative(path, current_path);
    imageLoaded[i] = ImageLoader::load_rgba8(std::filesystem::path(images[i]));
    imageLoadedArray[i] = imageLoaded[i].get();
    pathArray.push_back(relative_path.string());
  }
  Core::GUID img_guid = guid;
  Core::ORID img_orid = Core::requestORID();
  Core::ResourceManager::get()->database.registerResource(img_orid, img_guid);
  GFX::GFXManager::get()->registerTextureResourceCubemap(img_guid,
                                                         imageLoadedArray);
  GFX::Texture* texture =
      Core::ResourceManager::get()->getResource<GFX::Texture>(img_guid);
  texture->orid = img_orid;
  texture->guid = img_guid;
  texture->resourcePathArray = pathArray;
  texture->serialize();
}

auto GFXManager::registerTextureResourceCubemap(
    Core::GUID guid,
    std::array<Image::Image<Image::COLOR_R8G8B8A8_UINT>*, 6> images) noexcept
    -> void {
  GFX::Texture textureResource = {};
  RHI::BufferDescriptor stagingBufferDescriptor;
  stagingBufferDescriptor.size = images[0]->data.size * 6;
  stagingBufferDescriptor.usage = (uint32_t)RHI::BufferUsage::COPY_SRC;
  stagingBufferDescriptor.memoryProperties =
      (uint32_t)RHI::MemoryProperty::HOST_VISIBLE_BIT |
      (uint32_t)RHI::MemoryProperty::HOST_COHERENT_BIT;
  stagingBufferDescriptor.mappedAtCreation = true;
  std::unique_ptr<RHI::Buffer> stagingBuffer =
      rhiLayer->getDevice()->createBuffer(stagingBufferDescriptor);
  std::future<bool> mapped =
      stagingBuffer->mapAsync(0, 0, stagingBufferDescriptor.size);
  if (mapped.get()) {
    char* mapdata =
        (char*)stagingBuffer->getMappedRange(0, stagingBufferDescriptor.size);
    uint32_t offset = 0;
    memcpy((void*)&(mapdata[offset]), images[0]->data.data,
           (size_t)images[0]->data.size);
    offset += images[0]->data.size;
    memcpy((void*)&(mapdata[offset]), images[1]->data.data,
           (size_t)images[1]->data.size);
    offset += images[1]->data.size;
    memcpy((void*)&(mapdata[offset]), images[2]->data.data,
           (size_t)images[2]->data.size);
    offset += images[2]->data.size;
    memcpy((void*)&(mapdata[offset]), images[3]->data.data,
           (size_t)images[3]->data.size);
    offset += images[3]->data.size;
    memcpy((void*)&(mapdata[offset]), images[4]->data.data,
           (size_t)images[4]->data.size);
    offset += images[4]->data.size;
    memcpy((void*)&(mapdata[offset]), images[5]->data.data,
           (size_t)images[5]->data.size);
    offset += images[5]->data.size;
    stagingBuffer->unmap();
  }
  std::unique_ptr<RHI::CommandEncoder> commandEncoder =
      rhiLayer->getDevice()->createCommandEncoder({nullptr});
  // create texture image
  textureResource.texture =
      rhiLayer->getDevice()->createTexture(RHI::TextureDescriptor{
          {(uint32_t)images[0]->width, (uint32_t)images[0]->height, 6},
          1,
          1,
          1,
          RHI::TextureDimension::TEX2D,
          RHI::TextureFormat::RGBA8_UNORM,
          (uint32_t)RHI::TextureUsage::COPY_DST |
              (uint32_t)RHI::TextureUsage::TEXTURE_BINDING,
          {RHI::TextureFormat::RGBA8_UNORM},
          RHI::TextureFlags::CUBE_COMPATIBLE,
      });

  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::TOP_OF_PIPE_BIT,
      (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {},
      {},
      {RHI::TextureMemoryBarrierDescriptor{
          textureResource.texture.get(),
          RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,
                                     1, 0, 6},
          (uint32_t)RHI::AccessFlagBits::NONE,
          (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
          RHI::TextureLayout::UNDEFINED,
          RHI::TextureLayout::TRANSFER_DST_OPTIMAL}}});

  commandEncoder->copyBufferToTexture(
      {0, 0, 0, stagingBuffer.get()},
      {textureResource.texture.get(),
       0,
       {},
       (uint32_t)RHI::TextureAspect::COLOR_BIT},
      {textureResource.texture->width(), textureResource.texture->height(), 6});

  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::TRANSFER_BIT,
      (uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {},
      {},
      {RHI::TextureMemoryBarrierDescriptor{
          textureResource.texture.get(),
          RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,
                                     1, 0, 1},
          (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
          (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
          RHI::TextureLayout::TRANSFER_DST_OPTIMAL,
          RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL}}});

  rhiLayer->getDevice()->getGraphicsQueue()->submit(
      {commandEncoder->finish({})});
  rhiLayer->getDevice()->getGraphicsQueue()->waitIdle();
  textureResource.originalView =
      textureResource.texture->createView(RHI::TextureViewDescriptor{
          RHI::TextureFormat::RGBA8_UNORM, RHI::TextureViewDimension::CUBE,
          (uint32_t)RHI::TextureAspect::COLOR_BIT, 0, 1, 0, 6});
  textureResource.guid = guid;
  Core::ResourceManager::get()->addResource(guid, std::move(textureResource));
}

auto GFXManager::registerTextureResource(
    Core::GUID guid, RHI::TextureDescriptor const& desc) noexcept -> void {
  GFX::Texture textureResource = {};
  // create texture image
  textureResource.texture = rhiLayer->getDevice()->createTexture(desc);
  // transition layout
  RHI::TextureAspectFlags aspectMask = 0;
  RHI::TextureLayout targetLayout = {};
  RHI::AccessFlags targetAccessFlags = {};
  if (desc.usage & (uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT) {
    aspectMask |= (uint32_t)RHI::TextureAspect::COLOR_BIT;
    targetLayout = RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL;
    targetAccessFlags =
        (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_READ_BIT |
        (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT;
  } else if (desc.usage & (uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT) {
    aspectMask |= (uint32_t)RHI::TextureAspect::DEPTH_BIT;
    targetLayout = RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL;
    targetAccessFlags =
        (uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT |
        (uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  } else if (desc.usage & (uint32_t)RHI::TextureUsage::TEXTURE_BINDING) {
    aspectMask |= (uint32_t)RHI::TextureAspect::COLOR_BIT;
    targetLayout = RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL;
    targetAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT;
  } else if (desc.usage & (uint32_t)RHI::TextureUsage::STORAGE_BINDING) {
    aspectMask |= (uint32_t)RHI::TextureAspect::COLOR_BIT;
    targetLayout = RHI::TextureLayout::GENERAL;
    targetAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;
  }else if (desc.usage & (uint32_t)RHI::TextureUsage::COPY_DST) {
    aspectMask |= (uint32_t)RHI::TextureAspect::COLOR_BIT;
    targetLayout = RHI::TextureLayout::TRANSFER_DST_OPTIMAL;
    targetAccessFlags = (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT;
  }
  // do transition commands
  std::unique_ptr<RHI::CommandEncoder> commandEncoder =
      rhiLayer->getDevice()->createCommandEncoder({nullptr});
  commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
      (uint32_t)RHI::PipelineStages::TOP_OF_PIPE_BIT,
      (uint32_t)RHI::PipelineStages::ALL_COMMANDS_BIT,
      (uint32_t)RHI::DependencyType::NONE,
      {}, {}, {RHI::TextureMemoryBarrierDescriptor{
          textureResource.texture.get(),
          RHI::ImageSubresourceRange{aspectMask, 0, desc.mipLevelCount, 0,
                                     uint32_t(desc.arrayLayerCount)},
          (uint32_t)RHI::AccessFlagBits::NONE, targetAccessFlags,
          RHI::TextureLayout::UNDEFINED, targetLayout}}});
  rhiLayer->getDevice()->getGraphicsQueue()->submit(
      {commandEncoder->finish({})});
  rhiLayer->getDevice()->getGraphicsQueue()->waitIdle();
  RHI::TextureViewDescriptor viewDesc = {desc.format};
  viewDesc.aspect = RHI::getTextureAspect(desc.format);
  viewDesc.baseArrayLayer = 0;
  viewDesc.arrayLayerCount = 1;
  viewDesc.dimension = RHI::TextureViewDimension::TEX2D;
  if (!hasBit(desc.flags,
              RHI::TextureFlags::CUBE_COMPATIBLE))  // if host visible we do not
                                                    // create view
    if (desc.size.depthOrArrayLayers != 1) {
      //textureResource.viewArrays.resize(6);
      //for (int i = 0; i < desc.size.depthOrArrayLayers; ++i) {
      //  viewDesc.baseArrayLayer = i;
      //  textureResource.viewArrays[i] =
      //      textureResource.texture->createView(viewDesc);
      //}
      //viewDesc.dimension = RHI::TextureViewDimension::TEX2D_ARRAY;
      //viewDesc.baseArrayLayer = 0;
      //viewDesc.arrayLayerCount = desc.size.depthOrArrayLayers;
      //textureResource.originalView =
      //    textureResource.texture->createView(viewDesc);
    } else
      textureResource.originalView =
          textureResource.texture->createView(viewDesc);
  else {
    textureResource.viewArrays.resize(6);
    for (int i = 0; i < 6; ++i) {
      viewDesc.baseArrayLayer = i;
      textureResource.viewArrays[i] =
          textureResource.texture->createView(viewDesc);
    }
    viewDesc.dimension = RHI::TextureViewDimension::TEX2D_ARRAY;
    viewDesc.baseArrayLayer = 0;
    viewDesc.arrayLayerCount = 6;
    textureResource.originalView =
        textureResource.texture->createView(viewDesc);
  }

  textureResource.guid = guid;
  Core::ResourceManager::get()->addResource(guid, std::move(textureResource));
}

auto GFXManager::registerTextureResource(char const* filepath) noexcept
    -> Core::GUID {
  std::filesystem::path path(filepath);
  if (!std::filesystem::exists(path)) {
    path = "content/textures/lost.png";
  }
  if (path.extension() == ".dds") {
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path relative_path =
        std::filesystem::relative(path, current_path);
    std::unique_ptr<Image::Texture_Host> dds_tex = Image::DDS::fromDDS(path);
    Core::GUID img_guid =
        Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
    Core::ORID img_orid = Core::requestORID();
    Core::ResourceManager::get()->database.registerResource(img_orid, img_guid);
    GFX::GFXManager::get()->registerTextureResource(img_guid, dds_tex.get());
    GFX::Texture* texture =
        Core::ResourceManager::get()->getResource<GFX::Texture>(img_guid);
    texture->orid = img_orid;
    texture->guid = img_guid;
    uint32_t relative_length = relative_path.string().length();
    texture->resourcePath =
        (relative_length == 0) ? path.string() : relative_path.string();
    texture->serialize();
    texture->texture->setName("external_image::" +
                              texture->resourcePath.value());
    return img_guid;
  } else if (path.extension() == ".exr") {
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path relative_path =
        std::filesystem::relative(path, current_path);
    std::unique_ptr<Image::Image<Image::COLOR_R32G32B32A32_FLOAT>> img = ImageLoader::load_rgba32(path);
    if (img.get() == nullptr) return Core::INVALID_GUID;
    Core::GUID img_guid =
        Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
    Core::ORID img_orid = Core::requestORID();
    Core::ResourceManager::get()->database.registerResource(img_orid, img_guid);
    GFX::GFXManager::get()->registerTextureResource(img_guid, img.get());
    GFX::Texture* texture =
        Core::ResourceManager::get()->getResource<GFX::Texture>(img_guid);
    texture->orid = img_orid;
    texture->guid = img_guid;
    uint32_t relative_length = relative_path.string().length();
    texture->resourcePath =
        (relative_length == 0) ? path.string() : relative_path.string();
    texture->serialize();
    texture->texture->setName("external_image::" +
                              texture->resourcePath.value());
    return img_guid;
  } else {
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path relative_path =
        std::filesystem::relative(path, current_path);
    std::unique_ptr<Image::Image<Image::COLOR_R8G8B8A8_UINT>> img =
        ImageLoader::load_rgba8(path);
    Core::GUID img_guid =
        Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
    Core::ORID img_orid = Core::requestORID();
    Core::ResourceManager::get()->database.registerResource(img_orid, img_guid);
    GFX::GFXManager::get()->registerTextureResource(img_guid, img.get());
    GFX::Texture* texture =
        Core::ResourceManager::get()->getResource<GFX::Texture>(img_guid);
    texture->orid = img_orid;
    texture->guid = img_guid;
    uint32_t relative_length = relative_path.string().length();
    texture->resourcePath =
        (relative_length == 0) ? path.string() : relative_path.string();
    texture->serialize();
    texture->texture->setName("external_image::" +
                              texture->resourcePath.value());
    return img_guid;
  }
}

auto GFXManager::registerSamplerResource(
    Core::GUID guid, RHI::SamplerDescriptor const& desc) noexcept -> void {
  GFX::Sampler samplerResource = {};
  samplerResource.sampler = rhiLayer->getDevice()->createSampler(desc);
  Core::ResourceManager::get()->addResource(guid, std::move(samplerResource));
}

auto GFXManager::registerShaderModuleResource(
    Core::GUID guid, RHI::ShaderModuleDescriptor const& desc) noexcept -> void {
  GFX::ShaderModule shaderModuleResource = {};
  shaderModuleResource.shaderModule =
      rhiLayer->getDevice()->createShaderModule(desc);
  shaderModuleResource.reflection = SPIRV_TO_Reflection(desc.code, desc.stage);
  Core::ResourceManager::get()->addResource(guid,
                                            std::move(shaderModuleResource));
}

auto GFXManager::registerShaderModuleResource(
    Core::GUID guid, char const* filepath,
    RHI::ShaderModuleDescriptor const& desc) noexcept -> void {
  RHI::ShaderModuleDescriptor smDesc = desc;
  Core::Buffer buffer;
  Core::syncReadFile(std::filesystem::path(filepath).string().c_str(), buffer);
  smDesc.code = &buffer;
  GFX::ShaderModule shaderModuleResource = {};
  shaderModuleResource.shaderModule =
      rhiLayer->getDevice()->createShaderModule(smDesc);
  shaderModuleResource.reflection =
      SPIRV_TO_Reflection(smDesc.code, smDesc.stage);
  Core::ResourceManager::get()->addResource(guid,
                                            std::move(shaderModuleResource));
}

auto GFXManager::registerShaderModuleResource(
    char const* filepath, RHI::ShaderModuleDescriptor const& desc) noexcept
    -> Core::GUID {
  Core::GUID guid =
      Core::ResourceManager::get()->requestRuntimeGUID<GFX::ShaderModule>();
  registerShaderModuleResource(guid, filepath, desc);
  return guid;
}

auto GFXManager::requestOfflineTextureResource(Core::ORID orid) noexcept
    -> Core::GUID {
  if (orid == Core::INVALID_ORID) return Core::INVALID_GUID;
  Core::GUID guid = Core::ResourceManager::get()->database.findResource(orid);
  // if not loaded
  if (guid == Core::INVALID_GUID) {
    guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
    GFX::Texture texture;
    texture.deserialize(rhiLayer->getDevice(), orid);
    if (texture.resourcePath.has_value() == false ||
        !std::filesystem::exists(texture.resourcePath.value()))
    {
      std::unique_ptr<Image::Image<Image::COLOR_R8G8B8A8_UINT>> img =
          ImageLoader::load_rgba8(std::filesystem::path("content/textures/lost.png"));
      GFX::GFXManager::get()->registerTextureResource(guid, img.get());
    }
    else if (texture.resourcePath.value() == "USE_PATH_ARRAY") {
      GFX::GFXManager::get()->registerTextureResourceCubemap(
          guid, std::array<char const*, 6>{
                    texture.resourcePathArray.value()[0].c_str(),
                    texture.resourcePathArray.value()[1].c_str(),
                    texture.resourcePathArray.value()[2].c_str(),
                    texture.resourcePathArray.value()[3].c_str(),
                    texture.resourcePathArray.value()[4].c_str(),
                    texture.resourcePathArray.value()[5].c_str()});
    } else {
      std::filesystem::path filepath(texture.resourcePath.value());
      if (filepath.extension() == ".dds") {
        std::unique_ptr<Image::Texture_Host> dds_tex =
            Image::DDS::fromDDS(filepath);
        GFX::GFXManager::get()->registerTextureResource(guid, dds_tex.get());
      } else if (filepath.extension() == ".exr") {
        std::unique_ptr<Image::Image<Image::COLOR_R32G32B32A32_FLOAT>> img =
            ImageLoader::load_rgba32(
                std::filesystem::path(texture.resourcePath.value()));
        GFX::GFXManager::get()->registerTextureResource(guid, img.get());      
      } else {
        std::unique_ptr<Image::Image<Image::COLOR_R8G8B8A8_UINT>> img =
            ImageLoader::load_rgba8(
                std::filesystem::path(texture.resourcePath.value()));
        GFX::GFXManager::get()->registerTextureResource(guid, img.get());      
      }
    }
    Core::ResourceManager::get()->database.registerResource(orid, guid);
    GFX::Texture* texture_ptr =
        Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
    texture_ptr->orid = orid;
    texture_ptr->texture->setName(texture.name);
    texture_ptr->resourcePath = texture.resourcePath.has_value()
                                    ? texture.resourcePath.value()
                                    : "content/textures/lost.png";
    texture_ptr->resourcePathArray = texture.resourcePathArray;
    texture_ptr->name = texture.name;
  }
  return guid;
}

auto GFXManager::requestOfflineMeshResource(Core::ORID orid) noexcept
    -> Core::GUID {
  Core::GUID guid = Core::ResourceManager::get()->database.findResource(orid);
  // if not loaded
  if (guid == Core::INVALID_GUID) {
    guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Mesh>();
    GFX::Mesh mesh;
    mesh.deserialize(rhiLayer->getDevice(), orid);
    Core::ResourceManager::get()->database.registerResource(orid, guid);
    Core::ResourceManager::get()->addResource(guid, std::move(mesh));
  }
  return guid;
}

auto GFXManager::registerMaterialResource(char const* filepath) noexcept
    -> Core::GUID {
  std::filesystem::path path(filepath);
  std::filesystem::path current_path = std::filesystem::current_path();
  std::filesystem::path relative_path =
      std::filesystem::relative(path, current_path);
  Core::ORID orid =
      Core::ResourceManager::get()->database.findResourcePath(filepath);
  if (orid == Core::INVALID_ORID) {
    Core::GUID guid =
        Core::ResourceManager::get()->requestRuntimeGUID<GFX::Material>();
    orid = Core::ResourceManager::get()->database.mapResourcePath(filepath);
    Core::ResourceManager::get()->database.registerResource(orid, guid);
    GFX::Material material;
    material.path = std::string(filepath);
    material.loadPath();
    material.ORID = orid;
    material.serialize();
    Core::ResourceManager::get()->addResource(guid, std::move(material));
    return guid;
  } else {
    return requestOfflineMaterialResource(orid);
  }
}

auto GFXManager::requestOfflineMaterialResource(Core::ORID orid) noexcept
    -> Core::GUID {
  Core::GUID guid = Core::ResourceManager::get()->database.findResource(orid);
  // if not loaded
  if (guid == Core::INVALID_GUID) {
    guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Material>();
    GFX::Material material;
    material.deserialize(rhiLayer->getDevice(), orid);
    Core::ResourceManager::get()->database.registerResource(orid, guid);
    Core::ResourceManager::get()->addResource(guid, std::move(material));
  }
  return guid;
}

inline uint64_t hash(RHI::SamplerDescriptor const& desc) {
  uint64_t hashed_value = 0;
  hashed_value |= (uint64_t)(desc.addressModeU) << 62;
  hashed_value |= (uint64_t)(desc.addressModeV) << 60;
  hashed_value |= (uint64_t)(desc.addressModeW) << 58;
  hashed_value |= (uint64_t)(desc.magFilter) << 57;
  hashed_value |= (uint64_t)(desc.minFilter) << 56;
  hashed_value |= (uint64_t)(desc.mipmapFilter) << 55;
  hashed_value |= (uint64_t)(desc.compare) << 50;
  return hashed_value;
}

auto GFXManager::GlobalSamplerTable::fetch(
    RHI::SamplerDescriptor const& desc) noexcept -> RHI::Sampler* {
  const uint64_t id = hash(desc);
  auto find = hash_samplers.find(id);
  if (find == hash_samplers.end()) {
    Core::GUID guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
    GFX::GFXManager::get()->registerSamplerResource(guid, desc);
    GFX::Sampler* sampler = Core::ResourceManager::get()->getResource<GFX::Sampler>(guid);
    hash_samplers[id] = sampler->sampler.get();
    return sampler->sampler.get();
  } else {
    return find->second;
  }
}

auto GFXManager::GlobalSamplerTable::fetch(
    RHI::AddressMode address, RHI::FilterMode filter,
    RHI::MipmapFilterMode mipmap) noexcept -> RHI::Sampler* {
  RHI::SamplerDescriptor desc;
  desc.addressModeU = address;
  desc.addressModeV = address;
  desc.addressModeW = address;
  desc.magFilter = filter;
  desc.minFilter = filter;
  desc.mipmapFilter = mipmap;
  return fetch(desc);
}


//
//auto GFXManager::registerDefualtSamplers() noexcept -> void {
//  // create all default samplers
//  commonSampler.defaultSampler =
//      Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
//  registerSamplerResource(commonSampler.defaultSampler,
//                          RHI::SamplerDescriptor{
//                              RHI::AddressMode::REPEAT,
//                              RHI::AddressMode::REPEAT,
//                              RHI::AddressMode::REPEAT,
//                              RHI::FilterMode::LINEAR,
//                              RHI::FilterMode::LINEAR
//                          });
//  commonSampler.clamp_nearest =
//      Core::ResourceManager::get()->requestRuntimeGUID<GFX::Sampler>();
//  registerSamplerResource(GFX::GFXManager::get()->commonSampler.clamp_nearest,
//                          RHI::SamplerDescriptor{});
//  Core::ResourceManager::get()
//      ->getResource<GFX::Sampler>(
//          GFX::GFXManager::get()->commonSampler.defaultSampler)
//      ->sampler->setName("DefaultSampler");
//  Core::ResourceManager::get()
//      ->getResource<GFX::Sampler>(
//          GFX::GFXManager::get()->commonSampler.clamp_nearest)
//      ->sampler->setName("ClampNearestSampler");
//}

#pragma endregion
}