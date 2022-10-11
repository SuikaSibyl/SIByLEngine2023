module;
#include <memory>
#include <utility>
export module GFX.Resource:Mesh;
import RHI;

namespace SIByL::GFX
{
	export struct Mesh {
		/** ctors & rval copies */
		Mesh() = default;
		Mesh(Mesh&& mesh) = default;
		Mesh(Mesh const& mesh) = delete;
		auto operator=(Mesh && mesh) -> Mesh& = default;
		auto operator=(Mesh const& mesh) -> Mesh& = delete;
		/* vertex buffer layout */
		RHI::VertexBufferLayout vertexBufferLayout = {};
		/** primitive state */
		RHI::PrimitiveState primitiveState = {};
		/** the gpu vertex buffer */
		std::unique_ptr<RHI::Buffer> vertexBuffer = nullptr;
		/** the gpu index buffer */
		std::unique_ptr<RHI::Buffer> indexBuffer = nullptr;
	};
}