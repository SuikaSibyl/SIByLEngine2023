#pragma once
#include <set>
#include <vector>
#include <cstdint>
#include <bitset>
#include <queue>
#include <array>
#include <typeinfo>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <iterator>
#include <utility>
#include <functional>

#include <Print/SE.Core.Log.hpp>
#include <System/SE.Core.System.hpp>

namespace SIByL::Core {
SE_EXPORT template <int MAX_CAPACITY>
struct SparseSet {
  /** insert a new element into set */
  auto insert(int x) noexcept -> void;
  /** delete an element */
  auto remove(int x) noexcept -> void;
  /** search an element */
  auto search(int x) noexcept -> int;
  /** get the set of sparse index */
  auto getSparseSet() noexcept -> std::set<uint64_t>;
  /** sparse array */
  std::array<int, MAX_CAPACITY> sparse;
  /** dense array */
  std::array<int, MAX_CAPACITY> dense;
  /** element live in dense array */
  int livingElementCount = 0;
};

template <int MAX_CAPACITY>
auto SparseSet<MAX_CAPACITY>::insert(int x) noexcept -> void {
  if (x > MAX_CAPACITY) return;
  if (livingElementCount >= MAX_CAPACITY) return;
  if (search(x) != -1) return;
  dense[livingElementCount] = x;
  sparse[x] = livingElementCount;
  ++livingElementCount;
}

template <int MAX_CAPACITY>
auto SparseSet<MAX_CAPACITY>::remove(int x) noexcept -> void {
  if (search(x) == -1) return;
  int tmp = dense[livingElementCount - 1];
  dense[sparse[x]] = tmp;
  sparse[tmp] = sparse[x];
  --livingElementCount;
}

template <int MAX_CAPACITY>
auto SparseSet<MAX_CAPACITY>::search(int x) noexcept -> int {
  if (x > MAX_CAPACITY) return -1;
  if (sparse[x] < livingElementCount && dense[sparse[x]] == x) return sparse[x];
  return -1;
}

template <int MAX_CAPACITY>
auto SparseSet<MAX_CAPACITY>::getSparseSet() noexcept -> std::set<uint64_t> {
  return std::set<uint64_t>(dense.begin(), dense.begin() + livingElementCount);
}
/**
 * ECS System
 * @ref: https://austinmorlan.com/posts/entity_component_system
 * @ref: https://www.david-colson.com/2020/02/09/making-a-simple-ecs.html
 */

/** An entity handle is an alias of uint */
SE_EXPORT using EntityHandle = uint64_t;
/** An entity is a wrapper of entity handle */
struct Entity;
/** Max number of entities */
SE_EXPORT inline EntityHandle const constexpr MAX_ENTITIES = 65536;
/** Max number of entities */
SE_EXPORT inline EntityHandle const constexpr NULL_ENTITY =
    static_cast<EntityHandle>(-1);

/** Each component type has a unique ID given */
SE_EXPORT using ComponentType = uint8_t;
/** Max number of component types */
SE_EXPORT inline ComponentType const constexpr MAX_COMPONENTS = 64;

/** Store the component types an entity has */
SE_EXPORT using Signature = std::bitset<MAX_COMPONENTS>;

/** Manager to allocate and free entities */
SE_EXPORT struct EntityManager : public Manager {
  /** start up entity manager singleton */
  virtual auto startUp() noexcept -> void override;
  /** shut down the GFX manager */
  virtual auto shutDown() noexcept -> void override;
  /** get the singleton */
  static auto get() noexcept -> EntityManager*;
  /** create an entity */
  auto createEntity() noexcept -> Entity;
  /** destroy an entity */
  auto destroyEntity(EntityHandle id) noexcept -> void;
  /** get entity signature */
  auto getSignature(EntityHandle id) noexcept -> Signature&;

 private:
  /** entity manager singleton */
  static EntityManager* singleton;
  /** Available queue of entity IDs */
  std::queue<EntityHandle> availableEntities = {};
  /** signatures for all possible entities */
  std::array<Signature, MAX_ENTITIES> signatures = {};
  /** Number of living entities */
  uint64_t livingEntityCount = 0;
};

SE_EXPORT struct IComponentPool {
  /* virtual desctructor */
  virtual ~IComponentPool() = default;
  /** destroy an entity */
  virtual auto destroyEntity(EntityHandle entity) noexcept -> void = 0;
};

SE_EXPORT template <class T>
struct ComponentPool : public IComponentPool {
  /** get the component of an entity */
  auto getData(EntityHandle entity) noexcept -> T* {
    int denseId = componentSparseSet.search(entity);
    if (denseId == -1) return nullptr;
    return &componentPool[denseId];
  }
  /** insert the component of an entity */
  auto insertData(EntityHandle entity, T&& component) noexcept -> void {
    componentSparseSet.insert(entity);
    componentPool.push_back(std::move(component));
  }
  /** remove the component of an entity */
  auto removeData(EntityHandle entity) noexcept -> void {
    int denseId = componentSparseSet.search(entity);
    if (denseId != -1) {
      // remove the element
      int tmp =
          componentSparseSet.dense[componentSparseSet.livingElementCount - 1];
      componentSparseSet.dense[componentSparseSet.sparse[entity]] = tmp;
      componentPool[componentSparseSet.sparse[entity]] =
          std::move(componentPool[componentSparseSet.livingElementCount - 1]);
      componentSparseSet.sparse[tmp] = componentSparseSet.sparse[entity];
      --componentSparseSet.livingElementCount;
      componentPool.pop_back();
    }
  }
  /** destroy an entity */
  virtual auto destroyEntity(EntityHandle entity) noexcept -> void override {
    removeData(entity);
  }
  /** get corresponding entities */
  virtual auto getEntities() noexcept -> std::set<EntityHandle> {
    return componentSparseSet.getSparseSet();
  }

 private:
  /** components memory pool */
  std::vector<T> componentPool = {};
  /** component to find dense index for entities */
  SparseSet<MAX_ENTITIES> componentSparseSet = {};
};

SE_EXPORT template <class... Params>
struct View {
  std::vector<std::tuple<EntityHandle, Params&...>> internal_tuple;
  auto begin() { return internal_tuple.begin(); }
  auto end() { return internal_tuple.end(); }
};


SE_EXPORT struct ComponentSerializeEnv {
  std::unordered_map<uint64_t, uint64_t> const& mapper;
  std::vector<uint8_t> binary_buffer;
  /** Copy a buffer to the binary buffer and return the offset */
  auto push_to_buffer(void* data, size_t size) noexcept -> size_t {
    size_t const offset = binary_buffer.size();
    binary_buffer.resize(offset + size);
    memcpy(&binary_buffer[offset], data, size);
    return offset;
  }
  /** Copy a buffer from the binary buffer and return the offset */
  auto load_from_buffer(size_t offset, void* data, size_t size) const noexcept -> void {
    memcpy(data, &binary_buffer[offset], size); }
};

SE_EXPORT struct ComponentManager : public Manager {
  /** callback to each component serialize / deserialzie */
  using SerializeFn = std::function<void(void*, EntityHandle const&, ComponentSerializeEnv& env)>;
  using DeserializeFn = std::function<void(void*, EntityHandle const&, ComponentSerializeEnv const& env)>;
  /** start up component manager singleton */
  virtual auto startUp() noexcept -> void override;
  /** shut down the GFX manager */
  virtual auto shutDown() noexcept -> void override {
    componentTypes.clear();
    componentPools.clear();
    serializeFuncs.clear();
    deserializeFuncs.clear();
  }
  /** get the singleton */
  static auto get() noexcept -> ComponentManager*;
  /** register component */
  template <class T>
  inline auto registerComponent() noexcept -> void {
    char const* typeName = typeid(T).name();
    componentTypes[typeName] = nextComponentType;
    componentPools[typeName] = std::make_unique<ComponentPool<T>>();
    ++nextComponentType;
    SerializeFn serializeFunc = [](void* emitter, EntityHandle const& handle,
                                   ComponentSerializeEnv& env) -> void {
      std::invoke(&(T::serialize), emitter, handle, env);
    };
    serializeFuncs.push_back(serializeFunc);
    DeserializeFn deserializeFunc = [](void* emitter, EntityHandle const& handle,
           ComponentSerializeEnv const& env) -> void {
      std::invoke(&(T::deserialize), emitter, handle, env);
    };
    deserializeFuncs.push_back(deserializeFunc);
  }

  /** try serialize all registered components */
  inline auto trySerialize(void* emitter, EntityHandle const& handle,
    ComponentSerializeEnv& env) noexcept -> void {
    for (auto& func : serializeFuncs) func(emitter, handle, env);
  }
  /** try serialize all registered components */
  inline auto tryDeserialize(void* aos, EntityHandle const& handle,
                             ComponentSerializeEnv const& env) noexcept -> void {
    for (auto& func : deserializeFuncs) func(aos, handle, env);
  }
  /** get component type */
  template <class T>
  inline auto getComponentType() noexcept -> ComponentType {
    char const* typeName = typeid(T).name();
    if (componentTypes.find(typeName) == componentTypes.end())
      LogManager::Error("ECS :: Component Type not registered.");
    return componentTypes[typeName];
  }
  /** add component */
  template <class T>
  inline auto addComponent(EntityHandle entt, T&& component) noexcept -> void {
    this->getComponentPool<T>()->insertData(entt, std::move(component));
    EntityManager::get()->getSignature(entt).set(
        ComponentManager::get()->getComponentType<T>());
  }
  /** remove component */
  template <class T>
  inline auto removeComponent(EntityHandle entt) noexcept -> void {
    this->getComponentPool<T>()->removeData(entt);
  }
  /** get component */
  template <class T>
  inline auto getComponent(EntityHandle entt) noexcept -> T* {
    return this->getComponentPool<T>()->getData(entt);
  }
  /** remove entity */
  auto destroyEntity(EntityHandle entt) noexcept -> void {
    for (auto& iter : componentPools)
      if (EntityManager::get()->getSignature(entt).test(
              componentTypes[iter.first]))
        iter.second.get()->destroyEntity(entt);
  }
  /** get the view of multiple components */
  template <class... Params>
  auto view() noexcept -> View<Params...>;
  /** get component poo; */
  template <class T>
  auto getComponentPool() noexcept -> ComponentPool<T>* {
    char const* typeName = typeid(T).name();
    return static_cast<ComponentPool<T>*>(componentPools[typeName].get());
  }

 private:
  /* singleton */
  static ComponentManager* singleton;
  /** mapping name to component type */
  std::unordered_map<char const*, ComponentType> componentTypes = {};
  /** component pools */
  std::unordered_map<char const*, std::unique_ptr<IComponentPool>>
      componentPools = {};
  /** next component type */
  ComponentType nextComponentType = 0;
  /* */
  std::vector<SerializeFn> serializeFuncs;
  std::vector<DeserializeFn> deserializeFuncs;
};

SE_EXPORT struct Entity {
  /** constructor */
  Entity(EntityHandle const& handle) : handle(handle) {}
  /** add component to entity */
  template <typename T, typename... Args>
  auto addComponent(Args&&... args) noexcept -> T* {
    Core::ComponentManager::get()->addComponent<T>(
        handle, T(std::forward<Args>(args)...));
    return Core::ComponentManager::get()->getComponent<T>(handle);
  }
  /** get component from entity */
  template <typename T>
  auto getComponent() noexcept -> T* {
    return Core::ComponentManager::get()->getComponent<T>(handle);
  }

  // template<class T>
  // bool hasComponent() {
  //	return context->registry.all_of<T>(entityHandle);
  // }
  /** remove component from entity */
  template <class T>
  auto removeComponent() noexcept -> void {
    ComponentManager::get()->removeComponent<T>(handle);
  }

  EntityHandle handle;
};

template <class T>
auto initComponentOnRegister(Core::Entity& entity, T& component) noexcept -> bool {
  return true;
}

namespace Internal {
template <class T>
inline auto getEntitiesHaveComponent(ComponentManager* manager) noexcept
    -> std::set<EntityHandle> {
  return manager->getComponentPool<T>()->getEntities();
}

template <class T>
inline auto getEntitiesHaveComponents(ComponentManager* manager) noexcept
    -> std::set<EntityHandle> {
  return getEntitiesHaveComponent<T>(manager);
}

template <class T, class... Us>
  requires(sizeof...(Us) != 0)
inline auto getEntitiesHaveComponents(ComponentManager* manager) noexcept
    -> std::set<EntityHandle> {
  std::set<EntityHandle> set_x = getEntitiesHaveComponent<T>(manager);
  std::set<EntityHandle> set_y = getEntitiesHaveComponents<Us...>(manager);
  std::set<EntityHandle> set_r;
  std::set_intersection(set_x.begin(), set_x.end(), set_y.begin(), set_y.end(),
                        std::inserter(set_r, set_r.begin()));
  return set_r;
}

template <class T>
inline auto getComponentFromEntities(
    ComponentManager* manager, std::set<EntityHandle> const& entities) noexcept
    -> std::vector<T*> {
  std::vector<T*> components = {};
  for (auto entity : entities)
    components.emplace_back((manager->getComponent<T>(entity)));
  return components;
}

template <class T>
inline auto getComponentsFromEntities(
    ComponentManager* manager, std::set<EntityHandle> const& entities) noexcept
    -> std::vector<std::tuple<T&>> {
  std::vector<T*> components = getComponentFromEntities<T>(manager, entities);
  std::vector<std::tuple<T&>> tuples = {};
  for (auto comp : components) tuples.emplace_back(std::tuple<T&>{*comp});
  return tuples;
}

template <class T, class... Us>
  requires(sizeof...(Us) != 0)
inline auto getComponentsFromEntities(
    ComponentManager* manager, std::set<EntityHandle> const& entities) noexcept
    -> std::vector<std::tuple<T&, Us&...>> {
  std::vector<std::tuple<T&>> tuple_l =
      getComponentsFromEntities<T>(manager, entities);
  std::vector<std::tuple<Us&...>> tuple_r =
      getComponentsFromEntities<Us...>(manager, entities);
  std::vector<std::tuple<T&, Us&...>> tuples = {};
  for (int i = 0; i < entities.size(); ++i)
    tuples.emplace_back(std::tuple_cat(tuple_l[i], tuple_r[i]));
  return tuples;
}
}  // namespace Internal

template <class... Params>
auto ComponentManager::view() noexcept -> View<Params...> {
  std::set<EntityHandle> entities =
      Internal::getEntitiesHaveComponents<Params...>(this);
  View<Params...> view = {};
  std::vector<std::tuple<Params&...>> tuple_comp =
      Internal::getComponentsFromEntities<Params...>(this, entities);
  size_t idx = 0;
  for (auto entity : entities)
    view.internal_tuple.emplace_back(
        std::tuple_cat(std::make_tuple(entity), tuple_comp[idx++]));
  return view;
}
}  // namespace SIByL::Core