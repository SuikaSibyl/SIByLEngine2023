module;
#include <vector>
#include <memory>
#include <string>
#include <optional>
export module GFX.Resource:ShaderModule;
import Core.Resource.RuntimeManage;
import RHI;

namespace SIByL::GFX
{
	export struct ShaderModule :public Core::Resource {
		/** rhi shader module */
		std::unique_ptr<RHI::ShaderModule> shaderModule = nullptr;
	};
}