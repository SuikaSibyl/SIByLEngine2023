module;
#include <vector>
export module GFX.Components:MeshRenderer;
import GFX.Resource;

namespace SIByL::GFX
{
	struct MeshRenderer {
		/** materials in renderer */
		std::vector<Material*> materials = {};
	};
}