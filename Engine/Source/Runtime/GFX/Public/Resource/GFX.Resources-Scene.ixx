module;
#include <string>
#include <format>
#include <memory>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <yaml-cpp/node/node.h>
export module GFX.Resource:Scene;
import Core.ECS;
import Core.Memory;
import Core.IO;
import Core.Log;
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
		/** serialize scene */
		auto serialize(std::filesystem::path path) noexcept -> void;
		/** deserialize scene */
		auto deserialize(std::filesystem::path path) noexcept -> void;
		/** name description */
		std::string name = "new scene";
		/** mapping handle to GameObject */
		std::unordered_map<GameObjectHandle, GameObject> gameObjects;
		/** show wether the scene is modified */
		bool isDirty = false;
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

	auto Scene::serialize(std::filesystem::path path) noexcept -> void {
		YAML::Emitter out;
		out << YAML::BeginMap;
		// output name
		out << YAML::Key << "SceneName" << YAML::Value << name;
		// output nodes
		out << YAML::Key << "SceneNodes" << YAML::Value << YAML::BeginSeq;
		for (auto iter = gameObjects.begin(); iter != gameObjects.end(); iter++) {

		}
		out << YAML::EndSeq;
		// output tail
		out << YAML::Key << "SceneEnd" << YAML::Value << "TRUE";
		out << YAML::EndMap;
		Core::Buffer scene_proxy;
		scene_proxy.data = (void*)out.c_str();
		scene_proxy.size = out.size();
		Core::syncWriteFile(path, scene_proxy);
		scene_proxy.data = nullptr;
	}

	auto Scene::deserialize(std::filesystem::path path) noexcept -> void {
		gameObjects.clear();
		Core::Buffer scene_proxy;
		Core::syncReadFile(path, scene_proxy);
		YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(scene_proxy.data));
		// check scene name
		if (!data["SceneName"] || !data["SceneNodes"]) {
			Core::LogManager::Error(std::format("GFX :: Scene Name not found when deserializing {0}", path.string()));
		}
		auto scene_nodes = data["SceneNodes"];
		for (auto node : scene_nodes) {
			//uint64_t uid = node["uid"].as<uint64_t>();
			//uint64_t parent = node["parent"].as<uint64_t>();
			//auto components = node["components"];
			//auto tagComponent = components["TagComponent"].as<std::string>();
			//auto children = node["children"];
			//std::vector<uint64_t> children_uids(children.size());
			//uint32_t idx = 0;
			//if (children)
			//	for (auto child : children)
			//		children_uids[idx++] = child.as<uint64_t>();
			//tree.addNode(tagComponent, uid, parent, std::move(children_uids));

			//deserializeEntity(components, tree.nodes[uid].entity, asset_layer);

			//if (parent == 0)
			//	tree.appointRoot(uid);
		}
	}

#pragma endregion

}