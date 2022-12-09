module;
#include <vector>
export module SE.GFX.Core:MeshRenderer;
import SE.Core.ECS;
import :Material;
import :SerializeUtils;

namespace SIByL::GFX
{
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


#pragma region MESH_FILTER_COMPONENT_IMPL

	auto MeshRenderer::serialize(void* emitter, Core::EntityHandle const& handle) -> void {

	}

	auto MeshRenderer::deserialize(void* compAoS, Core::EntityHandle const& handle) -> void {

	}

#pragma endregion
}