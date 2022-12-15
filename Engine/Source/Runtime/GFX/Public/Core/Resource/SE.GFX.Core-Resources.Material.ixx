module;
#include <unordered_map>
#include <string>
export module SE.GFX.Core:Material;
import :Texture;
import SE.Core.Resource;

namespace SIByL::GFX
{
	export struct Material :public Core::Resource {
		std::unordered_map<std::string, Core::GUID> textures;
		/** get name */
		virtual auto getName() const noexcept -> char const* override { return name.c_str(); }
	private:
		/** resource name */
		std::string name = "New Material";
	};
}