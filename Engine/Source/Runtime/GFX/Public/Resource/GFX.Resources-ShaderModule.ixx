module;
#include <vector>
#include <memory>
#include <string>
#include <filesystem>
export module GFX.Resource:ShaderModule;
import Core.Resource.RuntimeManage;
import RHI;

namespace SIByL::GFX
{
	export struct ShaderModule :public Core::Resource {
		/** shader entry */
		struct Entry {
			RHI::ShaderStages stage;
			std::filesystem::path path;
		};
		/** all entries */
		std::vector<Entry> entries;
		/** rhi shader module */
		std::unique_ptr<ShaderModule> shaderModule = nullptr;
	};
}