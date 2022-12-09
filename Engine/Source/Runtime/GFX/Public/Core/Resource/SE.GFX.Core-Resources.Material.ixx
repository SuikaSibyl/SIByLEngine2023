module;
#include <unordered_map>
#include <string>
export module SE.GFX.Core:Material;
import :Texture;
import SE.Core.Resource;

namespace SIByL::GFX
{
	export struct Material {
		std::unordered_map<std::string, Core::GUID> textures;
	};
}