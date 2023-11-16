#pragma once
#include <functional>
#include <ECS/SE.Core.ECS.hpp>
#include <SE.Core.Reflect.hpp>
#include <SE.Core.Utility.hpp>

namespace SIByL {
struct IScriptBase {
  // Member fileds
  // ------------------------------------------------------
  Core::Entity entity = Core::NULL_ENTITY;
  // Virtual Mounted Functions
  // ------------------------------------------------------
  virtual auto OnCreate() noexcept -> void {}
  virtual auto OnDestroy() noexcept -> void {}
  virtual auto OnUpdate(double ts) noexcept -> void {}
  virtual auto RenderUI() noexcept -> void {}
  // Utils Functions
  // ------------------------------------------------------
  // Fetch the component of the entity
  template<class T>
  auto GetComponent() noexcept -> T* { return entity.getComponent<T>(); }
};

struct ScriptRegistry {
  // Member fileds
  // ------------------------------------------------------
  struct ScriptEntry { std::function<IScriptBase*()> instantiate; };
  std::unordered_map<std::string, ScriptEntry> registries;
  // Utils Functions
  // ------------------------------------------------------
  // Register the script entry by template class.
  template<class T> static auto RegisterScript() noexcept -> void {
    std::string const script_name = std::string(get_raw_name<ScriptRegistry>());
    std::unordered_map<std::string, ScriptEntry>& global_registries =
        Singleton<ScriptRegistry>::instance()->registries;
    auto iter = global_registries.find(script_name);
    if (iter == global_registries.end()) {
      ScriptEntry entry;
      entry.instantiate = []() { return new T(); };
      global_registries[script_name] = entry;
    } else {
      // Has already been registered.
    }
  }
  // Find the script register entry by class name.
  static auto FindEntry(std::string const& name) noexcept -> ScriptEntry*;
  // Find the script register entry by class name.
  template <class T> static auto FindEntry() noexcept -> ScriptEntry* {
    std::string const script_name = std::string(get_raw_name<ScriptRegistry>());
    std::unordered_map<std::string, ScriptEntry>& global_registries =
        Singleton<ScriptRegistry>::instance()->registries;
    auto iter = global_registries.find(script_name);
    if (iter == global_registries.end()) {
      return &(iter->second);
    } else {
      RegisterScript<T>();
      return &(global_registries[script_name]);
    }
  }
};

struct NativeScriptComponent {
  // Member fileds
  // ------------------------------------------------------'
  struct NativeScriptEntry {
    IScriptBase* instance = nullptr;
    ScriptRegistry::ScriptEntry* entry;
    std::string script_name;
    template <class T>
    auto bind() noexcept -> void {
      script_name = std::string(get_raw_name<ScriptRegistry>());
      entry = ScriptRegistry::FindEntry<T>();
    }
    auto bind(std::string const& name) noexcept -> void;
  };
  std::vector<NativeScriptEntry> entries;
  // Member Functions
  // ------------------------------------------------------
  auto addScript() noexcept -> NativeScriptEntry& { 
    entries.push_back({});
    return entries.back();
  }
  /** serialize */
  static auto serialize(void* emitter, Core::EntityHandle const& handle) -> void;
  /** deserialize */
  static auto deserialize(void* compAoS, Core::EntityHandle const& handle) -> void;
};
}