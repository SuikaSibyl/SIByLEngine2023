module;
#include <memory>
#include <utility>
#include <vector>
export module GFX.Resource:ASGroup;
import :Mesh;
import Core.Resource;
import RHI;

namespace SIByL::GFX
{
	export struct ASGroup {
		/** ctors & rval copies */
		ASGroup() = default;
		ASGroup(ASGroup&& group) = default;
		ASGroup(ASGroup const& group) = delete;
		auto operator=(ASGroup&& group)->ASGroup & = default;
		auto operator=(ASGroup const& group)->ASGroup & = delete;
		/** the gpu vertex buffer */
		std::unique_ptr<RHI::Buffer> GeometryInfoBuffer;
		std::unique_ptr<RHI::Buffer> vertexBufferArray;
		std::unique_ptr<RHI::Buffer> indexBufferArray;
		std::unique_ptr<RHI::TLAS> tlas = nullptr;
		/** the geometry info */
		struct GeometryInfo {
			uint32_t vertexOffset;
			uint32_t indexOffset;
		};
		std::vector<GeometryInfo> geometryInfo;
	};
}