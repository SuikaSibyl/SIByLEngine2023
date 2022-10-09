module;
#include <memory>
export module GFX.Resource:Mesh;
import RHI;

namespace SIByL::GFX
{
	export struct Mesh {
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