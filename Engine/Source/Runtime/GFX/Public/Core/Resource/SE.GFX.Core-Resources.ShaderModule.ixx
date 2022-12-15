module;
#include <vector>
#include <memory>
#include <string>
#include <optional>
export module SE.GFX.Core:ShaderModule;
import SE.Core.Resource;
import SE.RHI;

namespace SIByL::GFX
{
	export struct ShaderModule :public Core::Resource {
		/** rhi shader module */
		std::unique_ptr<RHI::ShaderModule> shaderModule = nullptr;
		/** get name */
		virtual auto getName() const noexcept -> char const* {
			return shaderModule->getName().c_str();
		}
	};
}