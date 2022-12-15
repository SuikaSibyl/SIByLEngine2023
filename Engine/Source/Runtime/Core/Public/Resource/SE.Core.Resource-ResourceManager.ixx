module;
#include <vector>
#include <cstdint>
#include <bitset>
#include <queue>
#include <array>
#include <typeinfo>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <memory>
#include <utility>
export module SE.Core.Resource:ResourceManager;
import :GUID;
import :OfflineManage;
import SE.Core.Log;
import SE.Core.System;

namespace SIByL::Core
{	
	/** Each resource type has a unique ID given */
	export using ResourceType = uint8_t;
	/** Max number of resource types */
	export inline ResourceType const constexpr MAX_RESOURCE_TYPES = 64;

	/** Interface of resources */
	export struct Resource {
		/** virtual destructor */
		virtual ~Resource() = default;
		/** get name */
		virtual auto getName() const noexcept -> char const* = 0;
	};

	export struct IResourcePool {
		/** virtual destructor */
		virtual ~IResourcePool() = default;
		/** get all resources */
		virtual auto getAllGUID() const noexcept -> std::vector<GUID> const& = 0;
		/** get resource name */
		virtual auto getResourceName(GUID guid) const noexcept -> char const* = 0;
	};

	export template<class T> struct ResourcePool :public IResourcePool {
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
		auto insertResource(GUID guid, T && resource) noexcept -> void {
			resourcePool.insert({ guid, std::move(resource) });
			GUIDs.push_back(guid);
		}
		/** remove the resource */
		auto removeData(GUID guid) noexcept -> void {
			resourcePool.erase(guid);
			for (auto iter = GUIDs.begin(); iter != GUIDs.end(); iter++) {
				if (*iter == guid){
					GUIDs.erase(iter);
					break;
				}
			}
		}
		/** getPool */
		auto getPool() const noexcept -> std::unordered_map<GUID, T> const& {
			return resourcePool;
		}
		/** get all resources */
		virtual auto getAllGUID() const noexcept -> std::vector<GUID> const& override {
			return GUIDs;
		}
		/** get resource name */
		virtual auto getResourceName(GUID guid) const noexcept -> char const* override {
			T const* resource = getResource(guid);
			if (resource == nullptr) return nullptr;
			else return resource->getName();
		}
	private:
		/** resource memory pool */
		std::unordered_map<GUID, T> resourcePool = {};
		std::vector<GUID> GUIDs;
	};

	/** Manager to allocate and free resources */
	export struct ResourceManager :public Manager {
		/** start up resource manager singleton */
		virtual auto startUp() noexcept -> void override;
		/** get the singleton */
		static auto get() noexcept -> ResourceManager*;
		/** clear all resources */
		auto clear() noexcept -> void;
		/** register resource */
		template <class T>
		auto registerResource() noexcept -> void {
			char const* typeName = typeid(T).name();
			resourceTypes.insert({ typeName, nextResourceType });
			resourcePools.insert({ typeName, std::make_unique<ResourcePool<T>>() });
			runtimeResourceCounters.insert({ typeName, 0 });
			++nextResourceType;
		}
		/** get resource type */
		template <class T>
		auto getResourceType() noexcept -> ResourceType {
			char const* typeName = typeid(T).name();
			if (resourceTypes.find(typeName) == resourceTypes.end())
				LogManager::Error("Resource :: Resource Type not registered.");
			return resourceTypes[typeName];
		}
		/** add resource */
		template <class T>
		auto addResource(GUID guid, T && resource) noexcept -> void {
			this->getResourcePool<T>()->insertResource(guid, std::move(resource));
		}
		/** remove resource */
		template <class T>
		auto removeResource(GUID guid) noexcept -> void {
			this->getResourcePool<T>()->removeResource(guid);
		}
		/** get resource */
		template <class T>
		auto getResource(GUID guid) noexcept -> T* {
			return this->getResourcePool<T>()->getResource(guid);
		}
		/** get resource */
		template <class T>
		auto requestRuntimeGUID() noexcept -> GUID {
			char const* typeName = typeid(T).name();
			std::string str = std::string(typeName) + std::to_string(runtimeResourceCounters[typeName]);
			++runtimeResourceCounters[typeName];
			return hashGUID(str);
		}
		/** get resource pools */
		auto getResourcePool() const noexcept -> std::unordered_map<char const*, std::unique_ptr<IResourcePool>> const& {
			return resourcePools;
		}
		/** get resource name */
		auto getResourceName(char const* typeName, GUID guid) noexcept -> char const* {
			auto iter = resourcePools.find(typeName);
			if (iter != resourcePools.end()) {
				return iter->second->getResourceName(guid);
			}
			return nullptr;
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
		std::unordered_map<char const*, std::unique_ptr<IResourcePool>> resourcePools = {};
		/** runtime resource counters */
		std::unordered_map<char const*, uint32_t> runtimeResourceCounters = {};
		/** next resource type */
		ResourceType nextResourceType = 0;
		/** get resource poo; */
		template <class T>
		auto getResourcePool() noexcept -> ResourcePool<T>* {
			char const* typeName = typeid(T).name();
			return static_cast<ResourcePool<T>*>(resourcePools[typeName].get());
		}
	};

#pragma region RESOURCE_MANAGER_IMPL

	ResourceManager* ResourceManager::singleton = nullptr;

	auto ResourceManager::startUp() noexcept -> void {
		singleton = this;
	}

	auto ResourceManager::get() noexcept -> ResourceManager* {
		return singleton;
	}

	auto ResourceManager::clear() noexcept -> void {
		resourcePools.clear();
	}

#pragma endregion

}