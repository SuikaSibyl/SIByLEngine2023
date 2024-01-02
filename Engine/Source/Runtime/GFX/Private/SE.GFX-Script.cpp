#include "../Public/SE.GFX-Script.hpp"
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/yaml.h>

namespace SIByL {
auto ScriptRegistry::FindEntry(std::string const& name) noexcept -> ScriptEntry* {
  std::unordered_map<std::string, ScriptEntry>& global_registries =
      Singleton<ScriptRegistry>::instance()->registries;
  auto iter = global_registries.find(name);
  if (iter == global_registries.end()) {
    return &(iter->second);
  } else {
    Core::LogManager::Error(
        "GFX :: Script entry not registred! *queried name: " + name);
    return nullptr;
  }
}

auto NativeScriptComponent::NativeScriptEntry::bind(
    std::string const& name) noexcept -> void {
  script_name = name;
  entry = ScriptRegistry::FindEntry(name);
}

auto NativeScriptComponent::serialize(void* pemitter, Core::EntityHandle const& handle,
    Core::ComponentSerializeEnv const& env) -> void {
  YAML::Emitter& emitter = *reinterpret_cast<YAML::Emitter*>(pemitter);
  Core::Entity entity(handle);
  NativeScriptComponent* native_script =
      entity.getComponent<NativeScriptComponent>();
  if (native_script != nullptr) {
    emitter << YAML::Key << "NativeScriptComponent";
    emitter << YAML::Value << YAML::BeginSeq;
    for (auto& iter : native_script->entries) 
      emitter << iter.script_name;
    emitter << YAML::EndSeq;
  }
}

auto NativeScriptComponent::deserialize(void* compAoS, Core::EntityHandle const& handle,
    Core::ComponentSerializeEnv const& env) -> void {
  YAML::NodeAoS& components = *reinterpret_cast<YAML::NodeAoS*>(compAoS);
  Core::Entity entity(handle);
  auto nsComponentAoS = components["NativeScriptComponent"];
  if (nsComponentAoS) {
    NativeScriptComponent* component =
        entity.addComponent<NativeScriptComponent>();
    for (auto child : nsComponentAoS) {
      NativeScriptEntry entry;
      entry.script_name = child.as<std::string>();
      entry.bind(entry.script_name);
      component->entries.push_back(entry);
    }
  }
}
}