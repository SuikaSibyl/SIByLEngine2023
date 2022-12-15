module;
#include <memory>
#include <utility>
#include <vector>
#include <string>
export module SE.GFX.Core:ASGroup;
import :Mesh;
import SE.Core.Resource;
import SE.Math.Geometric;
import SE.RHI;

namespace SIByL::GFX
{
	export struct ASGroup :public Core::Resource {
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
		std::unique_ptr<RHI::TLAS>	 tlas = nullptr;
		/** the geometry info */
		struct GeometryInfo {
			uint32_t vertexOffset;
			uint32_t indexOffset;
			uint32_t materialID;
			uint32_t padding = 0;
			Math::mat4 geometryTransform;
		};
		std::vector<GeometryInfo> geometryInfo;
		/** set name */
		virtual auto setName(std::string const& name) noexcept -> void { this->name = name; }
		/** get name */
		virtual auto getName() const noexcept -> char const* override { return name.c_str(); }
	private:
		std::string name = "New AsGroup";
	};
}