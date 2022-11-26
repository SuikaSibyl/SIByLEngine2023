module;
#include <cmath>
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
import Math.Trigonometric;
import RHI;
import GFX.SerializeUtils;

namespace SIByL::GFX
{
	export struct TagComponent {
		/** constructor */
		TagComponent(std::string const& name = "New GameObject") :name(name) {}
		// game object name
		std::string name;
		/** serialize */
		static auto serialize(void* emitter, Core::EntityHandle const& handle) -> void;
		/** deserialize */
		static auto deserialize(void* compAoS, Core::EntityHandle const& handle) -> void;
	};

#pragma region TAG_COMPONENT_IMPL

	auto TagComponent::serialize(void* pemitter, Core::EntityHandle const& handle) -> void {
		YAML::Emitter& emitter = *reinterpret_cast<YAML::Emitter*>(pemitter);
		Core::Entity entity(handle);
		TagComponent* tag = entity.getComponent<TagComponent>();
		if (tag != nullptr) {
			emitter << YAML::Key << "TagComponent";
			std::string const& name = tag->name;
			emitter << YAML::Value << name;
		}
	}

	auto TagComponent::deserialize(void* compAoS, Core::EntityHandle const& handle) -> void {
		YAML::NodeAoS& components = *reinterpret_cast<YAML::NodeAoS*>(compAoS);
		Core::Entity entity(handle);
		auto tagComponentAoS = components["TagComponent"];
		if (tagComponentAoS) {
			entity.getComponent<TagComponent>()->name = tagComponentAoS.as<std::string>();
		}
	}

#pragma endregion


	export struct TransformComponent {
		/** constructor */
		TransformComponent() = default;
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
		/** get transform */
		auto getTransform() noexcept -> Math::mat4;
		/** get rotated forward */
		auto getRotatedForward() noexcept -> Math::vec3 {
			Math::vec3 front;
			front.x = std::cos((eulerAngles.y)) * std::cos((eulerAngles.x));
			front.y = std::sin((eulerAngles.x));
			front.z = std::sin((eulerAngles.y)) * std::cos((eulerAngles.x));
			return front;
		}
		/** serialize */
		static auto serialize(void* emitter, Core::EntityHandle const& handle) -> void;
		/** deserialize */
		static auto deserialize(void* compAoS, Core::EntityHandle const& handle) -> void;
	};

#pragma region TRANSFORM_COMPONENT_IMPL
	
	auto TransformComponent::getTransform() noexcept -> Math::mat4 {
		return Math::mat4::translate(translation)
			* Math::Quaternion(eulerAngles).toMat4()
			* Math::mat4::scale(scale);
	}

	auto TransformComponent::serialize(void* pemitter, Core::EntityHandle const& handle) -> void {
		YAML::Emitter& emitter = *reinterpret_cast<YAML::Emitter*>(pemitter);
		Core::Entity entity(handle);
		TransformComponent* transform = entity.getComponent<TransformComponent>();
		if (transform != nullptr) {
			emitter << YAML::Key << "Transform";
			emitter << YAML::Value << YAML::BeginMap;
			emitter << YAML::Key << "translation" << YAML::Value << transform->translation;
			emitter << YAML::Key << "eulerAngles" << YAML::Value << transform->eulerAngles;
			emitter << YAML::Key << "scale" << YAML::Value << transform->scale;
			emitter << YAML::EndMap;
		}
	}

	auto TransformComponent::deserialize(void* compAoS, Core::EntityHandle const& handle) -> void {
		YAML::NodeAoS& components = *reinterpret_cast<YAML::NodeAoS*>(compAoS);
		Core::Entity entity(handle);
		auto transformComponentAoS = components["Transform"];
		if (transformComponentAoS) {
			TransformComponent* transform = entity.getComponent<TransformComponent>();
			Math::vec3 translation = transformComponentAoS["translation"].as<Math::vec3>();
			Math::vec3 eulerAngles = transformComponentAoS["eulerAngles"].as<Math::vec3>();
			Math::vec3 scale = transformComponentAoS["scale"].as<Math::vec3>();
			transform->translation = translation;
			transform->eulerAngles = eulerAngles;
			transform->scale = scale;
		}
	}

#pragma endregion

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
		Core::syncWriteFile(path, scene_proxy);
		scene_proxy.data = nullptr;
	}

	auto Scene::deserialize(std::filesystem::path path) noexcept -> void {
		//gameObjects.clear();
		Core::Buffer scene_proxy;
		Core::syncReadFile(path, scene_proxy);
		YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(scene_proxy.data));
		// check scene name
		if (!data["SceneName"] || !data["SceneNodes"]) {
			Core::LogManager::Error(std::format("GFX :: Scene Name not found when deserializing {0}", path.string()));
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
				for (auto child : children)
					go->children[idx++] = child.as<uint64_t>();
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

#pragma endregion

}