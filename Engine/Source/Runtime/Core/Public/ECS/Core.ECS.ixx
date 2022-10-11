module;
#include <vector>
#include <cstdint>
#include <bitset>
#include <queue>
#include <array>
#include <typeinfo>
#include <string>
#include <unordered_map>
#include <memory>
#include <utility>
export module Core.ECS;
import Core.Log;
import Core.System;
import Core.SparseSet;

namespace SIByL::Core
{
	/**
	* ECS System
	* @ref: https://austinmorlan.com/posts/entity_component_system
	* @ref: https://www.david-colson.com/2020/02/09/making-a-simple-ecs.html
	*/

	/** An entity handle is an alias of uint */
	export using EntityHandle = uint64_t;
	/** An entity is a wrapper of entity handle */
	struct Entity;
	/** Max number of entities */
	export inline EntityHandle const constexpr MAX_ENTITIES = 2048;
	/** Max number of entities */
	export inline EntityHandle const constexpr NULL_ENTITY = static_cast<EntityHandle>(-1);

	/** Each component type has a unique ID given */
	export using ComponentType = uint8_t;
	/** Max number of component types */
	export inline ComponentType const constexpr MAX_COMPONENTS = 64;

	/** Store the component types an entity has */
	export using Signature = std::bitset<MAX_COMPONENTS>;

	/** Manager to allocate and free entities */
	export struct EntityManager :public Manager {
		/** start up entity manager singleton */
		virtual auto startUp() noexcept -> void override;
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

	export struct IComponentPool {
		/** destroy an entity */
		virtual auto destroyEntity(EntityHandle entity) noexcept -> void = 0;
	};

	export template<class T> struct ComponentPool :public IComponentPool {
		/** get the component of an entity */
		auto getData(EntityHandle entity) noexcept -> T* {
			int denseId = componentSparseSet.search(entity);
			if (denseId == -1) return nullptr;
			return &componentPool[denseId];
		}
		/** insert the component of an entity */
		auto insertData(EntityHandle entity, T const& component) noexcept -> void {
			componentSparseSet.insert(entity);
			componentPool.push_back(component);
		}
		/** remove the component of an entity */
		auto removeData(EntityHandle entity) noexcept -> void {
			int denseId = componentSparseSet.search(entity);
			if (denseId != -1) {
				// remove the element
				int tmp = componentSparseSet.dense[componentSparseSet.livingElementCount - 1];
				componentSparseSet.dense[componentSparseSet.sparse[entity]] = tmp;
				componentPool[componentSparseSet.sparse[entity]] = componentPool[componentSparseSet.livingElementCount - 1];
				componentSparseSet.sparse[tmp] = componentSparseSet.sparse[entity];
				--componentSparseSet.livingElementCount;
				componentPool.pop_back();
			}
		}
		/** destroy an entity */
		virtual auto destroyEntity(EntityHandle entity) noexcept -> void override {
			removeData(entity);
		}
	private:
		/** components memory pool */
		std::vector<T> componentPool = {};
		/** component to find dense index for entities */
		SparseSet<MAX_ENTITIES> componentSparseSet = {};
	};

	export struct ComponentManager :public Manager {
		/** start up component manager singleton */
		virtual auto startUp() noexcept -> void override;
		/** get the singleton */
		static auto get() noexcept -> ComponentManager*;
		/** register component */
		template <class T>
		auto registerComponent() noexcept -> void {
			char const* typeName = typeid(T).name();
			componentTypes.insert({ typeName, nextComponentType });
			componentPools.insert({ typeName, std::make_unique<ComponentPool<T>>() });
			++nextComponentType;
		}
		/** get component type */
		template <class T>
		auto getComponentType() noexcept -> ComponentType {
			char const* typeName = typeid(T).name();
			if (componentTypes.find(typeName) == componentTypes.end())
				LogManager::Error("ECS :: Component Type not registered.");
			return componentTypes[typeName];
		}
		/** add component */
		template <class T>
		auto addComponent(EntityHandle entt, T const& component) noexcept -> void {
			this->getComponentPool<T>()->insertData(entt, component);
			EntityManager::get()->getSignature(entt).set(ComponentManager::get()->getComponentType<T>());
		}
		/** remove component */
		template <class T>
		auto removeComponent(EntityHandle entt) noexcept -> void {
			this->getComponentPool<T>()->removeData(entt);
		}
		/** get component */
		template <class T>
		auto getComponent(EntityHandle entt) noexcept -> T* {
			return this->getComponentPool<T>()->getData(entt);
		}
		/** remove entity */
		auto destroyEntity(EntityHandle entt) noexcept -> void {
			for (auto& iter : componentPools)
				if (EntityManager::get()->getSignature(entt).test(componentTypes[iter.first]))
					iter.second.get()->destroyEntity(entt);
		}
	private:
		/* singleton */
		static ComponentManager* singleton;
		/** mapping name to component type */
		std::unordered_map<char const*, ComponentType> componentTypes = {};
		/** component pools */
		std::unordered_map<char const*, std::unique_ptr<IComponentPool>> componentPools = {};
		/** next component type */
		ComponentType nextComponentType = 0;
		/** get component poo; */
		template <class T>
		auto getComponentPool() noexcept -> ComponentPool<T>* {
			char const* typeName = typeid(T).name();
			return static_cast<ComponentPool<T>*>(componentPools[typeName].get());
		}
	};

	export struct Entity {
		/** constructor */
		Entity(EntityHandle const& handle) :handle(handle) {}
		/** add component to entity */
		template<typename T, typename ... Args>
		auto addComponent(Args&&... args) noexcept -> T& {
			ComponentManager::get()->addComponent<T>(handle, T(std::forward<Args>(args)...));
			return *ComponentManager::get()->getComponent<T>(handle);
		}
		/** get component from entity */
		template<typename T>
		auto getComponent() noexcept -> T* {
			return Core::ComponentManager::get()->getComponent<T>(handle);
		}

		//template<class T>
		//bool hasComponent() {
		//	return context->registry.all_of<T>(entityHandle);
		//}
		/** remove component from entity */
		template<class T>
		auto removeComponent() noexcept -> void {
			ComponentManager::get()->removeComponent<T>(handle);
		}
		
		EntityHandle handle;
	};

#pragma region MISC_IMPL

	EntityManager* EntityManager::singleton = nullptr;
	ComponentManager* ComponentManager::singleton = nullptr;

	auto EntityManager::get() noexcept -> EntityManager* { return singleton; }
	auto ComponentManager::get() noexcept -> ComponentManager* { return singleton; }

	auto EntityManager::startUp() noexcept -> void {
		// initialize the queue with all possible entity IDs
		for (EntityHandle entity = 0; entity < MAX_ENTITIES; ++entity)
			availableEntities.emplace(entity);
		singleton = this;
	}
	auto ComponentManager::startUp() noexcept -> void {
		singleton = this;
	}

	auto EntityManager::createEntity() noexcept -> Entity {
		if (livingEntityCount >= MAX_ENTITIES) {
			LogManager::Error("ECS :: Entity creation exceed pool size.");
			return Entity(NULL_ENTITY);
		}
		// take an ID from the queue.
		++livingEntityCount;
		EntityHandle id = availableEntities.front();
		signatures[id] = 0;
		availableEntities.pop();
		return Entity(id);
	}

	auto EntityManager::destroyEntity(EntityHandle id) noexcept -> void {
		// push an ID back to the queue.
		--livingEntityCount;
		availableEntities.push(id);
		ComponentManager::get()->destroyEntity(id);
	}

	auto EntityManager::getSignature(EntityHandle id) noexcept -> Signature& {
		return signatures[id];
	}

#pragma endregion

}