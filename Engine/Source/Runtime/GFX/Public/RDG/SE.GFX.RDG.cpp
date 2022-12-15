module;
#include <string>
module SE.GFX.RDG;
import SE.RHI;

namespace SIByL::GFX
{
	auto RDGraph::createTexture(char const* name) noexcept -> RDGTexture* {
		return nullptr;
	}

	auto RDGraph::addPass(char const* pass_name, RDGPassFlag flag, CustomPassSetupFn const& custom_setup) noexcept -> RDGPassNode* {
		return nullptr;
	}

	auto RDGraph::execute() noexcept -> void {

	}
}