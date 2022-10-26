module;
#include <vector>
export module GFX.Components:MeshRenderer;
import GFX.Resource;

namespace SIByL::GFX
{
	export struct MeshRenderer {
		/** materials in renderer */
		std::vector<Material*> materials = {};
	};
}