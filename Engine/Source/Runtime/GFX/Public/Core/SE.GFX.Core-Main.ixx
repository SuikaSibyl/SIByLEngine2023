module;
#include <set>
#include <memory>
#include <vector>
#include <format>
#include <cstdint>
#include <utility>
#include <optional>
#include <string>
#include <typeinfo>
#include <filesystem>
#include <functional>
#include <unordered_map>
#include <yaml-cpp/yaml.h>
#include <yaml-cpp/node/node.h>
export module SE.GFX.Core:Main;
import :SerializeUtils;
import :GFXConfig;
import SE.Core.IO;
import SE.Core.Log;
import SE.Core.ECS;
import SE.Core.Memory;
import SE.Core.System;
import SE.Core.Resource;
import SE.Math.Misc;
import SE.Math.Geometric;
import SE.Image;
import SE.RHI;

namespace SIByL::GFX 
{
	struct Buffer;
	struct ShaderModule;
	struct Mesh;
	struct Texture;
	struct Sampler;

	/**
	* GFX Loader is a singleton sub-module of GFX Manager.
	* It manages how to load resources from
	*/
	export struct GFXLoader {

	};

	export struct Buffer :public Core::Resource {
		/** ctors & rval copies */
		Buffer() = default;
		Buffer(Buffer&& buffer) = default;
		Buffer(Buffer const& buffer) = delete;
		auto operator=(Buffer&& buffer)->Buffer & = default;
		auto operator=(Buffer const& buffer)->Buffer & = delete;
		/** the gpu vertex buffer */
		std::unique_ptr<RHI::Buffer> buffer = nullptr;
		/** get name */
		virtual auto getName() const noexcept -> char const* override { return buffer->getName().c_str(); }
	};

	export struct ShaderModule :public Core::Resource {
		/** rhi shader module */
		std::unique_ptr<RHI::ShaderModule> shaderModule = nullptr;
		/** get name */
		virtual auto getName() const noexcept -> char const* { return shaderModule->getName().c_str(); }
	};

	export struct Mesh :public Core::Resource {
		/** ctors & rval copies */
		Mesh() = default;
		Mesh(Mesh&& mesh) = default;
		Mesh(Mesh const& mesh) = delete;
		auto operator=(Mesh&& mesh)->Mesh & = default;
		auto operator=(Mesh const& mesh)->Mesh & = delete;
		/* vertex buffer layout */
		RHI::VertexBufferLayout vertexBufferLayout = {};
		/** primitive state */
		RHI::PrimitiveState primitiveState = {};
		/** the gpu|device vertex buffer */
		std::unique_ptr<RHI::Buffer> vertexBuffer_device = nullptr;
		std::unique_ptr<RHI::Buffer> positionBuffer_device = nullptr;
		std::unique_ptr<RHI::Buffer> indexBuffer_device = nullptr;
		/** the cpu|host vertex/index/position buffers */
		Core::Buffer	vertexBuffer_host = {};
		Core::Buffer	positionBuffer_host = {};
		Core::Buffer	indexBuffer_host = {};
		/** host-device copy */
		struct DeviceHostBufferInfo {
			uint32_t size = 0;
			bool onHost = false;
			bool onDevice = false;
		};
		DeviceHostBufferInfo vertexBufferInfo;
		DeviceHostBufferInfo positionBufferInfo;
		DeviceHostBufferInfo indexBufferInfo;
		/** binded ORID */
		Core::ORID ORID = Core::ORID_NONE;
		/** submeshes */
		struct Submesh {
			uint32_t offset;
			uint32_t size;
			uint32_t baseVertex;
			uint32_t matID;
		};
		std::vector<Submesh> submeshes;
		/** resource name */
		std::string name = "New Mesh";
		/** serialize */
		inline auto serialize() noexcept -> void;
		/** deserialize */
		inline auto deserialize(RHI::Device* device, Core::ORID orid) noexcept -> void;
		/** get name */
		virtual auto getName() const noexcept -> char const* {
			return name.c_str();
		}
	};

	export struct Texture :public Core::Resource {
		/** ctors & rval copies */
		Texture() = default;
		Texture(Texture&& texture) = default;
		Texture(Texture const& texture) = delete;
		auto operator=(Texture&& texture)->Texture & = default;
		auto operator=(Texture const& texture)->Texture & = delete;
		/** serialize */
		inline auto serialize() noexcept -> void;
		/** deserialize */
		inline auto deserialize(RHI::Device* device, Core::ORID orid) noexcept -> void;
		/** resrouce GUID */
		Core::GUID guid;
		/** resrouce ORID */
		Core::ORID orid = Core::ORID_NONE;
		/** texture */
		std::unique_ptr<RHI::Texture> texture = nullptr;
		/** texture display view*/
		std::unique_ptr<RHI::TextureView> originalView = nullptr;
		/** path string */
		std::optional<std::string> resourcePath;
		/** name */
		std::string name;
		/** get name */
		virtual auto getName() const noexcept -> char const* {
			return texture->getName().c_str();
		}
	};

	export struct Sampler :public Core::Resource {
		/** ctors & rval copies */
		Sampler() = default;
		Sampler(Sampler&& sampler) = default;
		Sampler(Sampler const& sampler) = delete;
		auto operator=(Sampler&& sampler)->Sampler & = default;
		auto operator=(Sampler const& sampler)->Sampler & = delete;
		/* rhi sampler */
		std::unique_ptr<RHI::Sampler> sampler = nullptr;
		/** get name */
		virtual auto getName() const noexcept -> char const* {
			return sampler->getName().c_str();
		}
	};
	/** Material Template */
	export struct MaterialTemplate {
		/** add a const data entry to the template */
		inline auto addConstantData(std::string const& name, RHI::DataFormat format) noexcept -> MaterialTemplate&;
		/** add a texture entry to the template */
		inline auto addTexture(std::string const& name) noexcept -> MaterialTemplate&;
		/** const data entries */
		std::unordered_map<std::string, RHI::DataFormat> constDataEntries;
		/** texture entries */
		std::vector<std::string> textureEntries;
	};

	/** Material */
	export struct Material :public Core::Resource {
		/** add a const data entry to the template */
		inline auto addConstantData(std::string const& name, RHI::DataFormat format) noexcept -> Material&;
		/** add a texture entry to the template */
		inline auto addTexture(std::string const& name, Core::GUID guid) noexcept -> Material&;
		/** register from a template */
		inline auto registerFromTemplate(MaterialTemplate const& mat_template) noexcept -> void;
		/** get name */
		virtual auto getName() const noexcept -> char const* override { return name.c_str(); }
		/** serialize */
		inline auto serialize() noexcept -> void;
		/** deserialize */
		inline auto deserialize(RHI::Device* device, Core::ORID orid) noexcept -> void;
		/** all textures in material */
		std::unordered_map<std::string, Core::GUID> textures;
		/** ORID of the material */
		Core::ORID ORID = Core::ORID_NONE;
		/** resource name */
		std::string name = "New Material";
		/** resource path */
		std::string path;
	};

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
		auto getRotatedForward() const noexcept -> Math::vec3 {
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

	export struct Scene :public Core::Resource {
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
		/** get name */
		virtual auto getName() const noexcept -> char const* override {
			return name.c_str();
		}
	};

	export struct CameraComponent {
		auto getViewMat() noexcept -> Math::mat4;
		auto getProjectionMat() noexcept -> Math::mat4;

		enum struct ProjectType {
			PERSPECTIVE,
			ORTHOGONAL,
		};

		/** serialize */
		static auto serialize(void* emitter, Core::EntityHandle const& handle) -> void;
		/** deserialize */
		static auto deserialize(void* compAoS, Core::EntityHandle const& handle) -> void;

		float fovy = 45.f;
		float aspect = 1;
		float near = 0.1f;
		float far = 100.0f;

		float left_right = 0;
		float bottom_top = 0;
		ProjectType projectType = ProjectType::PERSPECTIVE;
		bool isPrimaryCamera = false;

	private:
		Math::mat4 view;
		Math::mat4 projection;
	};

	export struct MeshReference {
		/* constructor */
		MeshReference() = default;
		/** mesh */
		Mesh* mesh = nullptr;
		/** serialize */
		static auto serialize(void* emitter, Core::EntityHandle const& handle) -> void;
		/** deserialize */
		static auto deserialize(void* compAoS, Core::EntityHandle const& handle) -> void;
	};

	export struct MeshRenderer {
		/* constructor */
		MeshRenderer() = default;
		/** materials in renderer */
		std::vector<Material*> materials = {};
		/** serialize */
		static auto serialize(void* emitter, Core::EntityHandle const& handle) -> void;
		/** deserialize */
		static auto deserialize(void* compAoS, Core::EntityHandle const& handle) -> void;
	};

	/** A singleton manager manages graphic components and resources. */
	export struct GFXManager :public Core::Manager {
		// online resource registers
		/** create / register online buffer resource */
		auto registerBufferResource(Core::GUID guid, RHI::BufferDescriptor const& desc) noexcept -> void;
		/** create / register online texture resource */
		auto registerTextureResource(Core::GUID guid, RHI::TextureDescriptor const& desc) noexcept -> void;
		auto registerTextureResource(Core::GUID guid, Image::Image<Image::COLOR_R8G8B8A8_UINT>* image) noexcept -> void;
		auto registerTextureResource(char const* filepath) noexcept -> Core::GUID;
		/** create / register online sampler resource */
		auto registerSamplerResource(Core::GUID guid, RHI::SamplerDescriptor const& desc) noexcept -> void;
		/** create / register online shader resource */
		auto registerShaderModuleResource(Core::GUID guid, RHI::ShaderModuleDescriptor const& desc) noexcept -> void;
		auto registerShaderModuleResource(Core::GUID guid, char const* filepath, RHI::ShaderModuleDescriptor const& desc) noexcept -> void;
		// ofline resource request
		/** request offline texture resource */
		auto requestOfflineTextureResource(Core::ORID orid) noexcept -> Core::GUID;
		/** request offline mesh resource */
		auto requestOfflineMeshResource(Core::ORID orid) noexcept -> Core::GUID;
		/** request offline material resource */
		auto requestOfflineMaterialResource(Core::ORID orid) noexcept -> Core::GUID;
		/** RHI layer */
		RHI::RHILayer* rhiLayer = nullptr;
		/** common samplers */
		struct CommonSampler {
			Core::GUID defaultSampler;
		} commonSampler;
		/** config singleton */
		GFXConfig config = {};
		/** start up the GFX manager */
		virtual auto startUp() noexcept -> void override;
		/** shut down the GFX manager */
		virtual auto shutDown() noexcept -> void override;
		/* get singleton */
		static inline auto get() noexcept -> GFXManager* { return singleton; }
	private:
		/** singleton */
		static GFXManager* singleton;
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

#pragma region MESH_IMPL

	inline auto Mesh::serialize() noexcept -> void {
		if (ORID == Core::ORID_NONE) {
			ORID = Core::requestORID();
		}
		std::filesystem::path metadata_path = "./bin/" + std::to_string(ORID) + ".meta";
		std::filesystem::path bindata_path = "./bin/" + std::to_string(ORID) + ".bin";
		// handle metadata
		{
			YAML::Emitter out;
			out << YAML::BeginMap;
			// output type
			out << YAML::Key << "ResourceType" << YAML::Value << "Mesh";
			out << YAML::Key << "Name" << YAML::Value << name;
			out << YAML::Key << "ORID" << YAML::Value << ORID;
			// output VertexBufferLayout
			out << YAML::Key << "VertexBufferLayout" << YAML::Value;
			out << YAML::BeginMap;
			{
				out << YAML::Key << "ArrayStride" << YAML::Value << vertexBufferLayout.arrayStride;
				out << YAML::Key << "VertexStepMode" << YAML::Value << (uint32_t)vertexBufferLayout.stepMode;
				out << YAML::Key << "VertexAttributes" << YAML::Value << YAML::BeginSeq;
				for (int i = 0; i < vertexBufferLayout.attributes.size(); i++) {
					out << YAML::BeginMap;
					out << YAML::Key << "VertexFormat" << YAML::Value << (uint32_t)vertexBufferLayout.attributes[i].format;
					out << YAML::Key << "Offset" << YAML::Value << vertexBufferLayout.attributes[i].offset;
					out << YAML::Key << "Location" << YAML::Value << vertexBufferLayout.attributes[i].shaderLocation;
					out << YAML::EndMap;
				}
				out << YAML::EndSeq;
			}
			out << YAML::EndMap;
			// output PrimitiveState
			out << YAML::Key << "PrimitiveState" << YAML::Value;
			out << YAML::BeginMap;
			{
				out << YAML::Key << "PrimitiveTopology" << YAML::Value << (uint32_t)primitiveState.topology;
				out << YAML::Key << "IndexFormat" << YAML::Value << (uint32_t)primitiveState.stripIndexFormat;
				out << YAML::Key << "FrontFace" << YAML::Value << (uint32_t)primitiveState.frontFace;
				out << YAML::Key << "CullMode" << YAML::Value << (uint32_t)primitiveState.cullMode;
				out << YAML::Key << "UnclippedDepth" << YAML::Value << primitiveState.unclippedDepth;
			}
			out << YAML::EndMap;

			out << YAML::Key << "VertexBufferSize" << YAML::Value << vertexBufferInfo.size;
			out << YAML::Key << "IndexBufferSize" << YAML::Value << indexBufferInfo.size;
			out << YAML::Key << "PosOnlyBufferSize" << YAML::Value << positionBufferInfo.size;
			// output submeshes
			out << YAML::Key << "Submeshes" << YAML::Value << YAML::BeginSeq;
			for (int i = 0; i < submeshes.size(); i++) {
				out << YAML::BeginMap;
				out << YAML::Key << "BaseVertex" << YAML::Value << submeshes[i].baseVertex;
				out << YAML::Key << "Offset" << YAML::Value << submeshes[i].offset;
				out << YAML::Key << "Size" << YAML::Value << submeshes[i].size;
				out << YAML::Key << "MatID" << YAML::Value << submeshes[i].matID;
				out << YAML::EndMap;
			}
			out << YAML::EndSeq;
			// output tail
			out << YAML::Key << "End" << YAML::Value << "TRUE";
			out << YAML::EndMap;
			Core::Buffer scene_proxy;
			scene_proxy.data = (void*)out.c_str();
			scene_proxy.size = out.size();
			Core::syncWriteFile(metadata_path, scene_proxy);
			scene_proxy.data = nullptr;
		}
		// handle binary data
		int vbsize = vertexBufferInfo.size;
		int ibsize = indexBufferInfo.size;
		int pbsize = positionBufferInfo.size;
		Core::Buffer mergedBuffer(vbsize + ibsize + pbsize);
		if (vertexBufferInfo.onHost) {
			memcpy(mergedBuffer.data, vertexBuffer_host.data, vbsize);
		}
		else if (vertexBufferInfo.onDevice) {
			vertexBuffer_device->getDevice()->waitIdle();
			vertexBuffer_device->getDevice()->readbackDeviceLocalBuffer(vertexBuffer_device.get(), mergedBuffer.data, vbsize);
		}
		if (indexBufferInfo.onHost) {
			memcpy(&(((char*)(mergedBuffer.data))[vbsize]), indexBuffer_host.data, ibsize);
		}
		else if (indexBufferInfo.onDevice) {
			indexBuffer_device->getDevice()->waitIdle();
			indexBuffer_device->getDevice()->readbackDeviceLocalBuffer(indexBuffer_device.get(), &(((char*)(mergedBuffer.data))[vbsize]), ibsize);
		}
		if (positionBufferInfo.onHost) {
			memcpy(&(((char*)(mergedBuffer.data))[vbsize + ibsize]), positionBuffer_host.data, pbsize);
		}
		else if (positionBufferInfo.onDevice) {
			positionBuffer_device->getDevice()->waitIdle();
			positionBuffer_device->getDevice()->readbackDeviceLocalBuffer(positionBuffer_device.get(), &(((char*)(mergedBuffer.data))[vbsize + ibsize]), pbsize);
		}

		Core::syncWriteFile(bindata_path, mergedBuffer);
	}

	inline auto Mesh::deserialize(RHI::Device* device, Core::ORID orid) noexcept -> void {
		ORID = orid;
		std::filesystem::path metadata_path = "./bin/" + std::to_string(ORID) + ".meta";
		std::filesystem::path bindata_path = "./bin/" + std::to_string(ORID) + ".bin";

		//gameObjects.clear();
		Core::Buffer metadata;
		Core::syncReadFile(metadata_path, metadata);
		YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(metadata.data));
		// check scene name
		if (data["ResourceType"].as<std::string>() != "Mesh") {
			Core::LogManager::Error(std::format("GFX :: Mesh resource not found when deserializing, ORID: {0}", std::to_string(orid)));
			return;
		}
		name = data["Name"].as<std::string>();
		auto vbbl_node = data["VertexBufferLayout"];
		vertexBufferLayout.arrayStride = vbbl_node["ArrayStride"].as<size_t>();
		vertexBufferLayout.stepMode = (RHI::VertexStepMode)vbbl_node["ArrayStride"].as<uint32_t>();
		auto attribute_nodes = vbbl_node["VertexAttributes"];
		for (auto node : attribute_nodes) {
			RHI::VertexAttribute attribute;
			attribute.format = (RHI::VertexFormat)node["VertexFormat"].as<uint32_t>();
			attribute.offset = node["Offset"].as<size_t>();
			attribute.shaderLocation = node["Location"].as<uint32_t>();
			vertexBufferLayout.attributes.push_back(attribute);
		}
		auto ps_node = data["PrimitiveState"];
		primitiveState.topology = (RHI::PrimitiveTopology)ps_node["PrimitiveTopology"].as<uint32_t>();
		primitiveState.stripIndexFormat = (RHI::IndexFormat)ps_node["IndexFormat"].as<uint32_t>();
		primitiveState.frontFace = (RHI::FrontFace)ps_node["FrontFace"].as<uint32_t>();
		primitiveState.cullMode = (RHI::CullMode)ps_node["CullMode"].as<uint32_t>();
		primitiveState.unclippedDepth = ps_node["UnclippedDepth"].as<bool>();
		// load buffers
		size_t vb_size, ib_size, pb_size;
		vb_size = data["VertexBufferSize"].as<size_t>();
		ib_size = data["IndexBufferSize"].as<size_t>();
		pb_size = data["PosOnlyBufferSize"].as<size_t>();
		vertexBufferInfo.size = vb_size;
		indexBufferInfo.size = ib_size;
		positionBufferInfo.size = pb_size;
		// load submeshes
		auto submeshes_node = data["Submeshes"];
		for (auto node : submeshes_node) {
			Submesh submesh;
			submesh.baseVertex = node["BaseVertex"].as<uint32_t>();
			submesh.offset = node["Offset"].as<uint32_t>();
			submesh.size = node["Size"].as<uint32_t>();
			submesh.matID = node["MatID"].as<uint32_t>();
			submeshes.push_back(submesh);
		}
		Core::Buffer bindata;
		Core::syncReadFile(bindata_path, bindata);

		MeshLoaderConfig meshConfig = GFXConfig::globalConfig->meshLoaderConfig;
		if (meshConfig.residentOnDevice) {
			vertexBufferInfo.onDevice = true;
			indexBufferInfo.onDevice = true;
			positionBufferInfo.onDevice = true;
			vertexBuffer_device = device->createDeviceLocalBuffer((void*)bindata.data, vb_size,
				(uint32_t)RHI::BufferUsage::VERTEX | (uint32_t)RHI::BufferUsage::STORAGE);
			indexBuffer_device = device->createDeviceLocalBuffer((void*)&(((uint16_t*)(&(((char*)(bindata.data))[vb_size])))[0]), ib_size,
				(uint32_t)RHI::BufferUsage::INDEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
				(uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
			if (pb_size != 0) {
				positionBuffer_device = device->createDeviceLocalBuffer((void*)&(((uint16_t*)(&(((char*)(bindata.data))[vb_size + ib_size])))[0]), pb_size,
					(uint32_t)RHI::BufferUsage::INDEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
					(uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
			}
		}
		else if (meshConfig.residentOnHost) {
			vertexBufferInfo.onHost = true;
			indexBufferInfo.onHost = true;
			positionBufferInfo.onHost = true;
			vertexBuffer_host = Core::Buffer(vb_size);
			indexBuffer_host = Core::Buffer(ib_size);
			positionBuffer_host = Core::Buffer(pb_size);
			memcpy(vertexBuffer_host.data, (void*)bindata.data, vb_size);
			memcpy(indexBuffer_host.data, (void*)&(((uint16_t*)(&(((char*)(bindata.data))[vb_size])))[0]), ib_size);
			memcpy(positionBuffer_host.data, (void*)&(((uint16_t*)(&(((char*)(bindata.data))[vb_size + ib_size])))[0]), pb_size);
		}
	}

#pragma endregion

#pragma region TEXTURE_IMPL

	inline auto Texture::serialize() noexcept -> void {
		// only serialize if has orid
		if (orid != Core::ORID_NONE && resourcePath.has_value()) {
			std::filesystem::path metadata_path = "./bin/" + std::to_string(orid) + ".meta";
			// handle metadata
			{	YAML::Emitter out;
				out << YAML::BeginMap;
				// output type
				out << YAML::Key << "ResourceType" << YAML::Value << "Texture";
				out << YAML::Key << "Name" << YAML::Value << getName();
				out << YAML::Key << "ORID" << YAML::Value << orid;
				out << YAML::Key << "path" << YAML::Value << resourcePath.value();
				out << YAML::Key << "End" << YAML::Value << "TRUE";
				out << YAML::EndMap;
				Core::Buffer tex_proxy;
				tex_proxy.data = (void*)out.c_str();
				tex_proxy.size = out.size();
				Core::syncWriteFile(metadata_path, tex_proxy);
				tex_proxy.data = nullptr;
			}
		}
	}

	inline auto Texture::deserialize(RHI::Device* device, Core::ORID ORID) noexcept -> void {
		orid = ORID;
		std::filesystem::path metadata_path = "./bin/" + std::to_string(ORID) + ".meta";
		Core::Buffer metadata;
		Core::syncReadFile(metadata_path, metadata);
		YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(metadata.data));
		// check scene name
		if (data["ResourceType"].as<std::string>() != "Texture") {
			Core::LogManager::Error(std::format("GFX :: Texture resource not found when deserializing, ORID: {0}", std::to_string(orid)));
			return;
		}
		name = data["Name"].as<std::string>();
		resourcePath = data["path"].as<std::string>();
	}

#pragma endregion

#pragma region MATERIAL_TEMPLATE_IMPL

	inline auto MaterialTemplate::addConstantData(std::string const& name, RHI::DataFormat format) noexcept -> MaterialTemplate& {
		constDataEntries[name] = format;
		return *this;
	}

	inline auto MaterialTemplate::addTexture(std::string const& name) noexcept -> MaterialTemplate& {
		textureEntries.push_back(name);
		return *this;
	}

#pragma endregion

#pragma region MATERIAL_IMPL

	inline auto Material::addConstantData(std::string const& name, RHI::DataFormat format) noexcept -> Material& {

		return *this;
	}

	inline auto Material::addTexture(std::string const& name, Core::GUID guid) noexcept -> Material& {
		textures[name] = guid;
		return *this;
	}

	inline auto Material::registerFromTemplate(MaterialTemplate const& mat_template) noexcept -> void {
		// add datas

		// add textures
		for (auto const& tex : mat_template.textureEntries)
			addTexture(tex, Core::INVALID_GUID);
	}

	inline auto Material::serialize() noexcept -> void {
		if (ORID == Core::ORID_NONE) {
			ORID = Core::requestORID();
		}
		if (path == "") {
			path = "./content/materials/" + std::to_string(ORID) + ".mat";
		}
		std::filesystem::path matdata_path = path;
		std::filesystem::path metadata_path = "./bin/" + std::to_string(ORID) + ".meta";
		// handle metadata
		{
			// handle metadata
			{	YAML::Emitter out;
			out << YAML::BeginMap;
			// output type
			out << YAML::Key << "ResourceType" << YAML::Value << "MaterialPtr";
			out << YAML::Key << "Name" << YAML::Value << getName();
			out << YAML::Key << "ORID" << YAML::Value << ORID;
			out << YAML::Key << "path" << YAML::Value << path;
			out << YAML::Key << "End" << YAML::Value << "TRUE";
			out << YAML::EndMap;
			Core::Buffer mat_proxy;
			mat_proxy.data = (void*)out.c_str();
			mat_proxy.size = out.size();
			Core::syncWriteFile(metadata_path, mat_proxy);
			mat_proxy.data = nullptr;
			}
		}
		// handle matdata
		{
			YAML::Emitter out;
			out << YAML::BeginMap;
			// output type
			out << YAML::Key << "ResourceType" << YAML::Value << "Material";
			out << YAML::Key << "Name" << YAML::Value << name;
			// output texture
			out << YAML::Key << "Textures" << YAML::Value;
			out << YAML::BeginSeq;
			{
				for (auto& [name, guid] : textures) {
					GFX::Texture* texture = Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
					out << YAML::BeginMap;
					out << YAML::Key << "Name" << YAML::Value << name;
					out << YAML::Key << "ORID" << YAML::Value << texture->orid;
					out << YAML::EndMap;
				}
			}
			out << YAML::EndSeq;
			// output tail
			out << YAML::Key << "End" << YAML::Value << "TRUE";
			out << YAML::EndMap;
			Core::Buffer mat_proxy;
			mat_proxy.data = (void*)out.c_str();
			mat_proxy.size = out.size();
			Core::syncWriteFile(matdata_path, mat_proxy);
			mat_proxy.data = nullptr;
		}
	}

	inline auto Material::deserialize(RHI::Device* device, Core::ORID orid) noexcept -> void {
		std::filesystem::path metadata_path = "./bin/" + std::to_string(orid) + ".meta";
		{
			Core::Buffer metadata;
			Core::syncReadFile(metadata_path, metadata);
			YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(metadata.data));
			// check scene name
			if (data["ResourceType"].as<std::string>() != "MaterialPtr") {
				Core::LogManager::Error(std::format("GFX :: MaterialPtr resource not found when deserializing, ORID: {0}", std::to_string(orid)));
				return;
			}
			name = data["Name"].as<std::string>();
			path = data["path"].as<std::string>();
		}
		// load data
		Core::Buffer matdata;
		Core::syncReadFile(path, matdata);
		YAML::NodeAoS data = YAML::Load(reinterpret_cast<char*>(matdata.data));
		// check resource type
		if (data["ResourceType"].as<std::string>() != "Material") {
			Core::LogManager::Error(std::format("GFX :: Material resource not found when deserializing, path: {0}", path));
			return;
		}
		auto texture_nodes = data["Textures"];
		for (auto node : texture_nodes) {
			std::string tex_name = node["Name"].as<std::string>();
			Core::ORID orid = node["ORID"].as<Core::ORID>();
			textures[tex_name] = GFXManager::get()->requestOfflineTextureResource(orid);
		}

		//name = data["Name"].as<std::string>();
		//auto vbbl_node = data["VertexBufferLayout"];
		//vertexBufferLayout.arrayStride = vbbl_node["ArrayStride"].as<size_t>();
		//vertexBufferLayout.stepMode = (RHI::VertexStepMode)vbbl_node["ArrayStride"].as<uint32_t>();
		//auto attribute_nodes = vbbl_node["VertexAttributes"];
		//for (auto node : attribute_nodes) {
		//	RHI::VertexAttribute attribute;
		//	attribute.format = (RHI::VertexFormat)node["VertexFormat"].as<uint32_t>();
		//	attribute.offset = node["Offset"].as<size_t>();
		//	attribute.shaderLocation = node["Location"].as<uint32_t>();
		//	vertexBufferLayout.attributes.push_back(attribute);
		//}
		//auto ps_node = data["PrimitiveState"];
		//primitiveState.topology = (RHI::PrimitiveTopology)ps_node["PrimitiveTopology"].as<uint32_t>();
		//primitiveState.stripIndexFormat = (RHI::IndexFormat)ps_node["IndexFormat"].as<uint32_t>();
		//primitiveState.frontFace = (RHI::FrontFace)ps_node["FrontFace"].as<uint32_t>();
		//primitiveState.cullMode = (RHI::CullMode)ps_node["CullMode"].as<uint32_t>();
		//primitiveState.unclippedDepth = ps_node["UnclippedDepth"].as<bool>();
		//// load buffers
		//size_t vb_size, ib_size, pb_size;
		//vb_size = data["VertexBufferSize"].as<size_t>();
		//ib_size = data["IndexBufferSize"].as<size_t>();
		//pb_size = data["PosOnlyBufferSize"].as<size_t>();
		//// load submeshes
		//auto submeshes_node = data["Submeshes"];
		//for (auto node : submeshes_node) {
		//	Submesh submesh;
		//	submesh.baseVertex = node["BaseVertex"].as<uint32_t>();
		//	submesh.offset = node["Offset"].as<uint32_t>();
		//	submesh.size = node["Size"].as<uint32_t>();
		//	submesh.matID = node["MatID"].as<uint32_t>();
		//	submeshes.push_back(submesh);
		//}
		//Core::Buffer bindata;
		//Core::syncReadFile(bindata_path, bindata);

		//vertexBuffer_device = device->createDeviceLocalBuffer((void*)bindata.data, vb_size,
		//	(uint32_t)RHI::BufferUsage::VERTEX | (uint32_t)RHI::BufferUsage::STORAGE);
		//indexBuffer_device = device->createDeviceLocalBuffer((void*)&(((uint16_t*)(&(((char*)(bindata.data))[vb_size])))[0]), ib_size,
		//	(uint32_t)RHI::BufferUsage::INDEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
		//	(uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
		//if (pb_size != 0) {
		//	positionBuffer_device = device->createDeviceLocalBuffer((void*)&(((uint16_t*)(&(((char*)(bindata.data))[vb_size + ib_size])))[0]), pb_size,
		//		(uint32_t)RHI::BufferUsage::INDEX | (uint32_t)RHI::BufferUsage::SHADER_DEVICE_ADDRESS |
		//		(uint32_t)RHI::BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY | (uint32_t)RHI::BufferUsage::STORAGE);
		//}
	}

#pragma endregion

#pragma region CAMERA_COMPONENT_IMPL

	auto CameraComponent::getViewMat() noexcept -> Math::mat4 {
		return view;
	}

	auto CameraComponent::getProjectionMat() noexcept -> Math::mat4 {
		if (projectType == ProjectType::PERSPECTIVE) {
			projection = Math::perspective(fovy, aspect, near, far).m;
		}
		else if (projectType == ProjectType::ORTHOGONAL) {
			projection = Math::orthographic(near, far).m;
		}
		return projection;
	}

	auto CameraComponent::serialize(void* pemitter, Core::EntityHandle const& handle) -> void {
		YAML::Emitter& emitter = *reinterpret_cast<YAML::Emitter*>(pemitter);
		Core::Entity entity(handle);
		CameraComponent* camera = entity.getComponent<CameraComponent>();
		if (camera != nullptr) {
			emitter << YAML::Key << "CameraComponent";
			emitter << YAML::Value << YAML::BeginMap;
			emitter << YAML::Key << "fovy" << YAML::Value << camera->fovy;
			emitter << YAML::Key << "aspect" << YAML::Value << camera->aspect;
			emitter << YAML::Key << "near" << YAML::Value << camera->near;
			emitter << YAML::Key << "far" << YAML::Value << camera->far;
			emitter << YAML::Key << "ProjectType" << YAML::Value << (uint32_t)camera->projectType;
			emitter << YAML::Key << "IsPrimary" << YAML::Value << camera->isPrimaryCamera;
			emitter << YAML::EndMap;
		}
	}

	auto CameraComponent::deserialize(void* compAoS, Core::EntityHandle const& handle) -> void {
		YAML::NodeAoS& components = *reinterpret_cast<YAML::NodeAoS*>(compAoS);
		Core::Entity entity(handle);
		auto cameraComponentAoS = components["CameraComponent"];
		if (cameraComponentAoS) {
			CameraComponent* camRef = entity.addComponent<CameraComponent>();
			camRef->fovy = cameraComponentAoS["fovy"].as<float>();
			camRef->aspect = cameraComponentAoS["aspect"].as<float>();
			camRef->near = cameraComponentAoS["near"].as<float>();
			camRef->far = cameraComponentAoS["far"].as<float>();
			camRef->projectType = (ProjectType)cameraComponentAoS["ProjectType"].as<uint32_t>();
			camRef->isPrimaryCamera = cameraComponentAoS["IsPrimary"].as<bool>();
		}
	}

#pragma endregion

#pragma region MESH_REFERENCE_COMPONENT_IMPL

	auto MeshReference::serialize(void* pemitter, Core::EntityHandle const& handle) -> void {
		YAML::Emitter& emitter = *reinterpret_cast<YAML::Emitter*>(pemitter);
		Core::Entity entity(handle);
		MeshReference* meshRef = entity.getComponent<MeshReference>();
		if (meshRef != nullptr) {
			emitter << YAML::Key << "MeshReference";
			emitter << YAML::Value << YAML::BeginMap;
			emitter << YAML::Key << "ORID" << YAML::Value << meshRef->mesh->ORID;
			emitter << YAML::EndMap;
		}
	}

	auto MeshReference::deserialize(void* compAoS, Core::EntityHandle const& handle) -> void {
		YAML::NodeAoS& components = *reinterpret_cast<YAML::NodeAoS*>(compAoS);
		Core::Entity entity(handle);
		auto meshRefComponentAoS = components["MeshReference"];
		if (meshRefComponentAoS) {
			MeshReference* meshRef = entity.addComponent<MeshReference>();
			Core::ORID orid = meshRefComponentAoS["ORID"].as<uint64_t>();
			Core::GUID guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Mesh>();
			GFX::Mesh mesh;
			Core::ResourceManager::get()->addResource(guid, std::move(mesh));
			Core::ResourceManager::get()->getResource<GFX::Mesh>(guid)->deserialize(RHI::RHILayer::get()->getDevice(), orid);
			meshRef->mesh = Core::ResourceManager::get()->getResource<GFX::Mesh>(guid);
		}
	}

#pragma endregion

#pragma region MESH_FILTER_COMPONENT_IMPL

	auto MeshRenderer::serialize(void* pemitter, Core::EntityHandle const& handle) -> void {
		//YAML::Emitter& emitter = *reinterpret_cast<YAML::Emitter*>(pemitter);
		//Core::Entity entity(handle);
		//CameraComponent* camera = entity.getComponent<CameraComponent>();
		//if (camera != nullptr) {
		//	emitter << YAML::Key << "MeshRendererComponent";
		//	emitter << YAML::Value << YAML::BeginMap;
		//	emitter << YAML::Key << "fovy" << YAML::Value << camera->fovy;
		//	emitter << YAML::Key << "aspect" << YAML::Value << camera->aspect;
		//	emitter << YAML::Key << "near" << YAML::Value << camera->near;
		//	emitter << YAML::Key << "far" << YAML::Value << camera->far;
		//	emitter << YAML::Key << "ProjectType" << YAML::Value << (uint32_t)camera->projectType;
		//	emitter << YAML::EndMap;
		//}
	}

	auto MeshRenderer::deserialize(void* compAoS, Core::EntityHandle const& handle) -> void {
		//YAML::NodeAoS& components = *reinterpret_cast<YAML::NodeAoS*>(compAoS);
		//Core::Entity entity(handle);
		//auto cameraComponentAoS = components["MeshRendererComponent"];
		//if (cameraComponentAoS) {
		//	CameraComponent* camRef = entity.addComponent<CameraComponent>();
		//	camRef->fovy = cameraComponentAoS["fovy"].as<float>();
		//	camRef->aspect = cameraComponentAoS["aspect"].as<float>();
		//	camRef->near = cameraComponentAoS["near"].as<float>();
		//	camRef->far = cameraComponentAoS["far"].as<float>();
		//	camRef->projectType = (ProjectType)cameraComponentAoS["ProjectType"].as<uint32_t>();
		//}
	}

#pragma endregion

#pragma region GFX_MANAGER_IMPL

	GFXManager* GFXManager::singleton = nullptr;

	auto GFXManager::startUp() noexcept -> void {
		// set singleton
		singleton = this;
		// register component types
		Core::ComponentManager::get()->registerComponent<GFX::TagComponent>();
		Core::ComponentManager::get()->registerComponent<GFX::TransformComponent>();
		Core::ComponentManager::get()->registerComponent<GFX::MeshReference>();
		Core::ComponentManager::get()->registerComponent<GFX::MeshRenderer>();
		Core::ComponentManager::get()->registerComponent<GFX::CameraComponent>();
		// register resource types
		Core::ResourceManager::get()->registerResource<GFX::Buffer>();
		Core::ResourceManager::get()->registerResource<GFX::Mesh>();
		Core::ResourceManager::get()->registerResource<GFX::Texture>();
		Core::ResourceManager::get()->registerResource<GFX::Sampler>();
		Core::ResourceManager::get()->registerResource<GFX::ShaderModule>();
		Core::ResourceManager::get()->registerResource<GFX::Material>();
		Core::ResourceManager::get()->registerResource<GFX::Scene>();
		// bind global config
		GFXConfig::globalConfig = &config;
	}

	auto GFXManager::shutDown() noexcept -> void {
		GFXConfig::globalConfig = nullptr;
	}

	auto GFXManager::registerBufferResource(Core::GUID guid, RHI::BufferDescriptor const& desc) noexcept -> void {
		GFX::Buffer bufferResource = {};
		bufferResource.buffer = rhiLayer->getDevice()->createBuffer(desc);
		Core::ResourceManager::get()->addResource(guid, std::move(bufferResource));
	}

	auto GFXManager::registerTextureResource(Core::GUID guid, Image::Image<Image::COLOR_R8G8B8A8_UINT>* image) noexcept -> void {
		GFX::Texture textureResource = {};
		RHI::BufferDescriptor stagingBufferDescriptor;
		stagingBufferDescriptor.size = image->data.size;
		stagingBufferDescriptor.usage = (uint32_t)RHI::BufferUsage::COPY_SRC;
		stagingBufferDescriptor.memoryProperties = (uint32_t)RHI::MemoryProperty::HOST_VISIBLE_BIT
			| (uint32_t)RHI::MemoryProperty::HOST_COHERENT_BIT;
		stagingBufferDescriptor.mappedAtCreation = true;
		std::unique_ptr<RHI::Buffer> stagingBuffer = rhiLayer->getDevice()->createBuffer(stagingBufferDescriptor);
		std::future<bool> mapped = stagingBuffer->mapAsync(0, 0, stagingBufferDescriptor.size);
		if (mapped.get()) {
			void* mapdata = stagingBuffer->getMappedRange(0, stagingBufferDescriptor.size);
			memcpy(mapdata, image->data.data, (size_t)stagingBufferDescriptor.size);
			stagingBuffer->unmap();
		}
		std::unique_ptr<RHI::CommandEncoder> commandEncoder = rhiLayer->getDevice()->createCommandEncoder({ nullptr });
		// create texture image
		textureResource.texture = rhiLayer->getDevice()->createTexture(RHI::TextureDescriptor{
			{(uint32_t)image->width,(uint32_t)image->height, 1},
			1,1,RHI::TextureDimension::TEX2D,
			RHI::TextureFormat::RGBA8_UNORM,
			(uint32_t)RHI::TextureUsage::COPY_DST | (uint32_t)RHI::TextureUsage::TEXTURE_BINDING,
			{ RHI::TextureFormat::RGBA8_UNORM }
			});

		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::TOP_OF_PIPE_BIT,
			(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				textureResource.texture.get(), RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::NONE,
				(uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
				RHI::TextureLayout::UNDEFINED,
				RHI::TextureLayout::TRANSFER_DST_OPTIMAL
			}}
			});

		commandEncoder->copyBufferToTexture(
			{ 0, 0, 0, stagingBuffer.get() },
			{ textureResource.texture.get(), 0, {}, (uint32_t)RHI::TextureAspect::COLOR_BIT },
			{ textureResource.texture->width(), textureResource.texture->height(), 1 });

		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
			(uint32_t)RHI::PipelineStages::FRAGMENT_SHADER_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				textureResource.texture.get(), RHI::ImageSubresourceRange{(uint32_t)RHI::TextureAspect::COLOR_BIT, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT,
				(uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT,
				RHI::TextureLayout::TRANSFER_DST_OPTIMAL,
				RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL
			}}
			});

		rhiLayer->getDevice()->getGraphicsQueue()->submit({ commandEncoder->finish({}) });
		rhiLayer->getDevice()->getGraphicsQueue()->waitIdle();
		textureResource.originalView = textureResource.texture->createView(RHI::TextureViewDescriptor{
			RHI::TextureFormat::RGBA8_UNORM });
		textureResource.guid = guid;
		Core::ResourceManager::get()->addResource(guid, std::move(textureResource));
	}

	auto GFXManager::registerTextureResource(Core::GUID guid, RHI::TextureDescriptor const& desc) noexcept -> void {
		GFX::Texture textureResource = {};
		// create texture image
		textureResource.texture = rhiLayer->getDevice()->createTexture(desc);
		// transition layout
		RHI::TextureAspectFlags aspectMask = 0;
		RHI::TextureLayout targetLayout = {};
		RHI::AccessFlags targetAccessFlags = {};
		if (desc.usage & (uint32_t)RHI::TextureUsage::COLOR_ATTACHMENT) {
			aspectMask |= (uint32_t)RHI::TextureAspect::COLOR_BIT;
			targetLayout = RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL;
			targetAccessFlags = (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_READ_BIT
				| (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT;
		}
		else if (desc.usage & (uint32_t)RHI::TextureUsage::DEPTH_ATTACHMENT) {
			aspectMask |= (uint32_t)RHI::TextureAspect::DEPTH_BIT;
			targetLayout = RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL;
			targetAccessFlags = (uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT
				| (uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		}
		else if (desc.usage & (uint32_t)RHI::TextureUsage::TEXTURE_BINDING) {
			aspectMask |= (uint32_t)RHI::TextureAspect::COLOR_BIT;
			targetLayout = RHI::TextureLayout::SHADER_READ_ONLY_OPTIMAL;
			targetAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT;
		}
		else if (desc.usage & (uint32_t)RHI::TextureUsage::COPY_DST) {
			aspectMask |= (uint32_t)RHI::TextureAspect::COLOR_BIT;
			targetLayout = RHI::TextureLayout::TRANSFER_DST_OPTIMAL;
			targetAccessFlags = (uint32_t)RHI::AccessFlagBits::TRANSFER_WRITE_BIT;
		}
		// do transition commands
		std::unique_ptr<RHI::CommandEncoder> commandEncoder = rhiLayer->getDevice()->createCommandEncoder({ nullptr });
		commandEncoder->pipelineBarrier(RHI::BarrierDescriptor{
			(uint32_t)RHI::PipelineStages::TOP_OF_PIPE_BIT,
			(uint32_t)RHI::PipelineStages::TRANSFER_BIT,
			(uint32_t)RHI::DependencyType::NONE,
			{}, {},
			{ RHI::TextureMemoryBarrierDescriptor{
				textureResource.texture.get(), RHI::ImageSubresourceRange{aspectMask, 0,1,0,1},
				(uint32_t)RHI::AccessFlagBits::NONE,
				targetAccessFlags,
				RHI::TextureLayout::UNDEFINED,
				targetLayout
			}}
			});
		rhiLayer->getDevice()->getGraphicsQueue()->submit({ commandEncoder->finish({}) });
		rhiLayer->getDevice()->getGraphicsQueue()->waitIdle();
		RHI::TextureViewDescriptor viewDesc = { desc.format };
		viewDesc.aspect = RHI::getTextureAspect(desc.format);
		if (!desc.hostVisible) // if host visible we do not create view
			textureResource.originalView = textureResource.texture->createView(viewDesc);
		textureResource.guid = guid;
		Core::ResourceManager::get()->addResource(guid, std::move(textureResource));
	}

	auto GFXManager::registerTextureResource(char const* filepath) noexcept -> Core::GUID {
		std::filesystem::path path(filepath);
		std::filesystem::path current_path = std::filesystem::current_path();
		std::filesystem::path relative_path = std::filesystem::relative(path, current_path);
		std::unique_ptr<Image::Image<Image::COLOR_R8G8B8A8_UINT>> img = ImageLoader::load_rgba8(std::filesystem::path(filepath));
		Core::GUID img_guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
		Core::ORID img_orid = Core::requestORID();
		GFX::GFXManager::get()->registerTextureResource(img_guid, img.get());
		GFX::Texture* texture = Core::ResourceManager::get()->getResource<GFX::Texture>(img_guid);
		texture->orid = img_orid;
		texture->guid = img_guid;
		texture->resourcePath = relative_path.string();
		return img_guid;
	}

	auto GFXManager::registerSamplerResource(Core::GUID guid, RHI::SamplerDescriptor const& desc) noexcept -> void {
		GFX::Sampler samplerResource = {};
		samplerResource.sampler = rhiLayer->getDevice()->createSampler(desc);
		Core::ResourceManager::get()->addResource(guid, std::move(samplerResource));
	}

	auto GFXManager::registerShaderModuleResource(Core::GUID guid, RHI::ShaderModuleDescriptor const& desc) noexcept -> void {
		GFX::ShaderModule shaderModuleResource = {};
		shaderModuleResource.shaderModule = rhiLayer->getDevice()->createShaderModule(desc);
		Core::ResourceManager::get()->addResource(guid, std::move(shaderModuleResource));
	}

	auto GFXManager::registerShaderModuleResource(Core::GUID guid, char const* filepath, RHI::ShaderModuleDescriptor const& desc) noexcept -> void {
		RHI::ShaderModuleDescriptor smDesc = desc;
		Core::Buffer buffer;
		Core::syncReadFile(std::filesystem::path(filepath), buffer);
		smDesc.code = &buffer;
		GFX::ShaderModule shaderModuleResource = {};
		shaderModuleResource.shaderModule = rhiLayer->getDevice()->createShaderModule(smDesc);
		Core::ResourceManager::get()->addResource(guid, std::move(shaderModuleResource));
	}

	auto GFXManager::requestOfflineTextureResource(Core::ORID orid) noexcept -> Core::GUID {
		Core::GUID guid = Core::ResourceManager::get()->database.findResource(orid);
		// if not loaded
		if (guid == Core::INVALID_GUID) {
			guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Texture>();
			GFX::Texture texture;
			texture.deserialize(rhiLayer->getDevice(), orid);
			std::unique_ptr<Image::Image<Image::COLOR_R8G8B8A8_UINT>> img = ImageLoader::load_rgba8(std::filesystem::path(texture.resourcePath.value()));
			GFX::GFXManager::get()->registerTextureResource(guid, img.get());
			Core::ResourceManager::get()->database.registerResource(orid, guid);
			GFX::Texture* texture_ptr = Core::ResourceManager::get()->getResource<GFX::Texture>(guid);
			texture_ptr->orid = orid;
			texture_ptr->resourcePath = texture.resourcePath.value();
			Core::ResourceManager::get()->addResource(guid, std::move(texture));
		}
		return guid;
	}

	auto GFXManager::requestOfflineMeshResource(Core::ORID orid) noexcept -> Core::GUID {
		Core::GUID guid = Core::ResourceManager::get()->database.findResource(orid);
		// if not loaded
		if (guid == Core::INVALID_GUID) {
			guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Mesh>();
			GFX::Mesh mesh;
			mesh.deserialize(rhiLayer->getDevice(), orid);
			Core::ResourceManager::get()->database.registerResource(orid, guid);
			Core::ResourceManager::get()->addResource(guid, std::move(mesh));
		}
		return guid;
	}
	
	auto GFXManager::requestOfflineMaterialResource(Core::ORID orid) noexcept -> Core::GUID {
		Core::GUID guid = Core::ResourceManager::get()->database.findResource(orid);
		// if not loaded
		if (guid == Core::INVALID_GUID) {
			guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Material>();
			GFX::Material material;
			material.deserialize(rhiLayer->getDevice(), orid);
			Core::ResourceManager::get()->database.registerResource(orid, guid);
			Core::ResourceManager::get()->addResource(guid, std::move(material));
		}
		return guid;
	}

#pragma endregion
}