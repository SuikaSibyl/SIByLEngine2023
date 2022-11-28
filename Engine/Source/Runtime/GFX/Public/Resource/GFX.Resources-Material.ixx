module;
#include <unordered_map>
#include <string>
export module GFX.Resource:Material;
import :Texture;
import Core.Resource;

namespace SIByL::GFX
{
	export struct Material {
		std::unordered_map<std::string, Core::GUID> textures;
	};
}