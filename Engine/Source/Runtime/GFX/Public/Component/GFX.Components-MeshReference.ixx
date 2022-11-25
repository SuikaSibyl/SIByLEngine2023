module;
#include <yaml-cpp/yaml.h>
#include <yaml-cpp/node/node.h>
export module GFX.Components:MeshReference;
import Core.ECS;
import Core.Resource;
import RHI;
import RHI.RHILayer;
import GFX.Resource;

namespace SIByL::GFX
{
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
			MeshReference& meshRef = entity.addComponent<MeshReference>();
			Core::ORID orid = meshRefComponentAoS["ORID"].as<uint64_t>();
			Core::GUID guid = Core::ResourceManager::get()->requestRuntimeGUID<GFX::Mesh>();
			GFX::Mesh mesh;
			Core::ResourceManager::get()->addResource(guid, std::move(mesh));
			Core::ResourceManager::get()->getResource<GFX::Mesh>(guid)->deserialize(RHI::RHILayer::get()->getDevice(), orid);
			meshRef.mesh = Core::ResourceManager::get()->getResource<GFX::Mesh>(guid);
		}
	}

#pragma endregion
}