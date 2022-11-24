export module GFX.Components:MeshReference;
import Core.ECS;
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

	auto MeshReference::serialize(void* emitter, Core::EntityHandle const& handle) -> void {

	}

	auto MeshReference::deserialize(void* compAoS, Core::EntityHandle const& handle) -> void {

	}

#pragma endregion
}