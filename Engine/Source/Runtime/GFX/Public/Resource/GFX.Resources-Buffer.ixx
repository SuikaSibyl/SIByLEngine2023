module;
#include <memory>
#include <utility>
export module GFX.Resource:Buffer;
import RHI;

namespace SIByL::GFX
{
	export struct Buffer {
		/** ctors & rval copies */
		Buffer() = default;
		Buffer(Buffer&& buffer) = default;
		Buffer(Buffer const& buffer) = delete;
		auto operator=(Buffer&& buffer)->Buffer & = default;
		auto operator=(Buffer const& buffer)->Buffer & = delete;
		/** the gpu vertex buffer */
		std::unique_ptr<RHI::Buffer> buffer = nullptr;
	};
}