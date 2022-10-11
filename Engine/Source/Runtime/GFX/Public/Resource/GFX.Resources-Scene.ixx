module;
#include <memory>
#include <vector>
#include <unordered_map>
export module GFX.Resource:Scene;
import Core.ECS;
import Math.Vector;
import Math.Matrix;
import Math.Transform;
import RHI;

namespace SIByL::GFX
{
	export struct TagComponent {
		// game object name
		std::string name;
	};

	export struct TransformComponent {
		/** decomposed transform - translation */
		Math::vec3 translation = { 0.0f, 0.0f, 0.0f };
		/** decomposed transform - eulerAngles */
		Math::vec3 eulerAngles = { 0.0f, 0.0f, 0.0f };
		/** decomposed transform - scale */
		Math::vec3 scale = { 1.0f, 1.0f, 1.0f };
		/** integrated world transform */
		Math::Transform transform = {};
		/** previous integrated world transform */
		Math::Transform previousTransform = {};
	};

	/** Game object handle is also the entity handle contained */
	export using GameObjectHandle = Core::EntityHandle;
	export GameObjectHandle NULL_GO = Core::NULL_ENTITY;
	/** Game object is a hierarchical wrapper of entity */
	export struct GameObject {
		auto getEntity() noexcept -> Core::Entity { return Core::Entity{ entity }; }
		GameObjectHandle parent = NULL_GO;
		Core::EntityHandle entity = Core::NULL_ENTITY;
		std::vector<GameObjectHandle> children = {};
	};

	export struct Scene {
		/** add a new entity */
		auto createGameObject(GameObjectHandle parent = NULL_GO) noexcept -> GameObjectHandle;
		/** remove an entity */
		auto removeGameObject(GameObjectHandle handle) noexcept -> void;
		/** get an game object */
		auto getGameObject(GameObjectHandle handle) noexcept -> GameObject*;
		/** move an game object */
		auto moveGameObject(GameObjectHandle handle) noexcept -> void;
		/** mapping handle to GameObject */
		std::unordered_map<GameObjectHandle, GameObject> gameObjects;
	};

#pragma region SCENE_IMPL

	auto Scene::createGameObject(GameObjectHandle parent) noexcept -> GameObjectHandle {
		Core::Entity entity = Core::EntityManager::get()->createEntity();
		gameObjects.insert({ entity.handle, GameObject{parent, entity.handle} });
		if (parent != NULL_GO) gameObjects[parent].children.push_back(entity.handle);
		gameObjects[entity.handle].getEntity().addComponent<TagComponent>("New GameObject");
		gameObjects[entity.handle].getEntity().addComponent<TransformComponent>();
		return GameObjectHandle(entity.handle);
	}

	auto Scene::removeGameObject(GameObjectHandle handle) noexcept -> void {
		if (gameObjects.find(handle) == gameObjects.end()) return;
		GameObject& go = gameObjects[handle];
		if (go.parent != NULL_GO) {
			// remove the go from its parent's children list
			GameObject& parent = gameObjects[go.parent];
			for (auto iter = parent.children.begin(); iter != parent.children.end(); ++iter) {
				if (*iter == handle) {
					parent.children.erase(iter);
					break;
				}
			}
		}
		// remove recursively its children
		std::vector<GameObjectHandle> children = go.children;
		for (auto child : children)
			removeGameObject(child);
		// remove the gameobject
		gameObjects.erase(handle);
		Core::EntityManager::get()->destroyEntity(handle);
	}

	auto Scene::getGameObject(GameObjectHandle handle) noexcept -> GameObject* {
		if (gameObjects.find(handle) == gameObjects.end()) return nullptr;
		else return &gameObjects[handle];
	}

#pragma endregion

}