#pragma once
#include <concepts>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include <System/SE.Core.System.hpp>
#include <Print/SE.Core.Log.hpp>

namespace SIByL::Core {
/** globally unique identifier */
SE_EXPORT using GUID = uint64_t;
SE_EXPORT constexpr inline GUID INVALID_GUID = uint64_t(-1);
/** Offline Resource ID */
SE_EXPORT using ORID = uint64_t;
SE_EXPORT constexpr inline ORID INVALID_ORID = uint64_t(-1);

/** hash a filesystem path to a GUID */
SE_EXPORT auto hashUID(char const* path) noexcept -> uint64_t;

/** request a orid from a global ORID generator */
SE_EXPORT auto requestORID() noexcept -> ORID;

/** resource database which could find active GUID for corresponding ORID */
SE_EXPORT struct ResourceDatabase {
  struct Entry {
    ORID orid;                 // offline resource id
    GUID guid = INVALID_ORID;  // runtime resource id
  };
  /** register a resourece to databse as loaded */
  auto registerResource(Core::ORID orid, Core::GUID guid) noexcept -> void;
  /** register a resourece to databse as loaded */
  auto findResource(Core::ORID orid) noexcept -> Core::GUID;
  /** find ORID for a path */
  auto findResourcePath(char const* path_c) noexcept -> Core::ORID;
  /** find ORID for a path or create one */
  auto mapResourcePath(char const* path_c) noexcept -> Core::ORID;
  /** serialize */
  auto serialize() noexcept -> void;
  /** deserialize */
  auto deserialize() noexcept -> void;

 private:  // --------------------------------------------------
  std::unordered_map<Core::ORID, Core::GUID> mapper;
  std::unordered_map<std::string, Core::ORID> resource_mapper;
};

/** Each resource type has a unique ID given */
SE_EXPORT using ResourceType = uint8_t;
/** Max number of resource types */
SE_EXPORT inline ResourceType const constexpr MAX_RESOURCE_TYPES = 64;

/** Interface of resources */
SE_EXPORT struct Resource {
  /** virtual destructor */
  virtual ~Resource() = default;
  /** get name */
  virtual auto getName() const noexcept -> char const* = 0;
};

/** SE_EXPORT resource type concpet */
SE_EXPORT template <class T>
concept StructResource = std::derived_from<T, Resource>;

SE_EXPORT struct IResourcePool {
  /** virtual destructor */
  virtual ~IResourcePool() = default;
  /** get all resources */
  virtual auto getAllGUID() const noexcept -> std::vector<GUID> const& = 0;
  /** get resource name */
  virtual auto getResourceName(GUID guid) const noexcept -> char const* = 0;
};

SE_EXPORT template <StructResource T>
struct ResourcePool : public IResourcePool {
  /** get the resource */
  auto getResource(GUID guid) noexcept -> T* {
    auto iter = resourcePool.find(guid);
    if (iter == resourcePool.end()) return nullptr;
    return &(iter->second);
  }
  /** get the resource */
  auto getResource(GUID guid) const noexcept -> T const* {
    auto iter = resourcePool.find(guid);
    if (iter == resourcePool.end()) return nullptr;
    return &(iter->second);
  }
  /** insert the resource */
  auto insertResource(GUID guid, T&& resource) noexcept -> void {
    resourcePool.insert({guid, std::move(resource)});
    GUIDs.push_back(guid);
  }
  /** remove the resource */
  auto removeResource(GUID guid) noexcept -> void {
    resourcePool.erase(guid);
    for (auto iter = GUIDs.begin(); iter != GUIDs.end(); iter++) {
      if (*iter == guid) {
        GUIDs.erase(iter);
        break;
      }
    }
  }
  /** getPool */
  auto getPool() const noexcept -> std::unordered_map<GUID, T> const& {
    return resourcePool;
  }
  auto getPool() noexcept -> std::unordered_map<GUID, T>& {
    return resourcePool;
  }
  /** get all resources */
  virtual auto getAllGUID() const noexcept
      -> std::vector<GUID> const& override {
    return GUIDs;
  }
  /** get resource name */
  virtual auto getResourceName(GUID guid) const noexcept
      -> char const* override {
    T const* resource = getResource(guid);
    if (resource == nullptr)
      return nullptr;
    else
      return resource->getName();
  }

 private:
  /** resource memory pool */
  std::unordered_map<GUID, T> resourcePool = {};
  std::vector<GUID> GUIDs;
};

/** Manager to allocate and free resources */
SE_EXPORT struct ResourceManager : public Manager {
  /** start up resource manager singleton */
  virtual auto startUp() noexcept -> void override;
  /* shut down the manager */
  virtual auto shutDown() noexcept -> void override;
  /** get the singleton */
  static auto get() noexcept -> ResourceManager*;
  /** clear all resources */
  auto clear() noexcept -> void;
  /** register resource */
  template <StructResource T>
  auto registerResource() noexcept -> void {
    char const* typeName = typeid(T).name();
    resourceTypes.insert({typeName, nextResourceType});
    resourcePools.insert({typeName, std::make_unique<ResourcePool<T>>()});
    runtimeResourceCounters.insert({typeName, 0});
    ++nextResourceType;
  }
  /** get resource type */
  template <StructResource T>
  auto getResourceType() noexcept -> ResourceType {
    char const* typeName = typeid(T).name();
    if (resourceTypes.find(typeName) == resourceTypes.end())
      LogManager::Error("Resource :: Resource Type not registered.");
    return resourceTypes[typeName];
  }
  /** add resource */
  template <StructResource T>
  auto addResource(GUID guid, T&& resource) noexcept -> void {
    this->getResourcePool<T>()->insertResource(guid, std::move(resource));
  }
  /** remove resource */
  template <StructResource T>
  auto removeResource(GUID guid) noexcept -> void {
    this->getResourcePool<T>()->removeResource(guid);
  }
  /** get resource */
  template <StructResource T>
  auto getResource(GUID guid) noexcept -> T* {
    return this->getResourcePool<T>()->getResource(guid);
  }
  /** get resource */
  template <StructResource T>
  auto requestRuntimeGUID() noexcept -> GUID {
    char const* typeName = typeid(T).name();
    std::string str = std::string(typeName) +
                      std::to_string(runtimeResourceCounters[typeName]);
    ++runtimeResourceCounters[typeName];
    return hashUID(str.c_str());
  }
  /** get resource pools */
  auto getResourcePool() const noexcept
      -> std::unordered_map<char const*,
                            std::unique_ptr<IResourcePool>> const& {
    return resourcePools;
  }
  /** get resource name */
  auto getResourceName(char const* typeName, GUID guid) noexcept
      -> char const* {
    auto iter = resourcePools.find(typeName);
    if (iter != resourcePools.end()) {
      return iter->second->getResourceName(guid);
    }
    return nullptr;
  }
  /** get resource poo; */
  template <StructResource T>
  auto getResourcePool() noexcept -> ResourcePool<T>* {
    char const* typeName = typeid(T).name();
    return static_cast<ResourcePool<T>*>(resourcePools[typeName].get());
  }
  /** resources registery */
  std::unordered_map<GUID, std::unique_ptr<Resource>> registry = {};
  /** resource database */
  ResourceDatabase database;

 private:
  /* singleton */
  static ResourceManager* singleton;
  /** mapping name to resource type */
  std::unordered_map<char const*, ResourceType> resourceTypes = {};
  /** resource pools */
  std::unordered_map<char const*, std::unique_ptr<IResourcePool>>
      resourcePools = {};
  /** runtime resource counters */
  std::unordered_map<char const*, uint32_t> runtimeResourceCounters = {};
  /** next resource type */
  ResourceType nextResourceType = 0;
};
}  // namespace SIByL::Core