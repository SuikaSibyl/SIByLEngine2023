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
export module Core.Resource.RuntimeManage:ResourceManager;
import :GUID;
import Core.Log;
import Core.System;

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
	};

	export struct IResourcePool {
		/** virtual destructor */
		virtual ~IResourcePool() = default;
	};

	export template<class T> struct ResourcePool :public IResourcePool {
		/** get the resource */
		auto getResource(GUID guid) noexcept -> T* {
			auto iter = resourcePool.find(guid);
			if (iter == resourcePool.end()) return nullptr;
			return &(iter->second);
		}
		/** insert the resource */
		auto insertResource(GUID guid, T && resource) noexcept -> void {
			resourcePool.insert({ guid, std::move(resource) });
		}
		/** remove the resource */
		auto removeData(GUID guid) noexcept -> void {
			resourcePool.erase(guid);
		}
	private:
		/** resource memory pool */
		std::unordered_map<GUID, T> resourcePool = {};
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
		/** resources registery */
		std::unordered_map<GUID, std::unique_ptr<Resource>> registry = {};
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