module;
#include <cmath>
#include <memory>
#include <string>
#include <cstdint>
export module SE.SRenderer.UDPTPipeline;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;
import SE.RDG;

import SE.SRenderer.UDPTPass;

namespace SIByL::SRP
{
	export struct UDPTPipeline :public RDG::Graph {

		UDPTPipeline() {
			addPass(std::make_unique<UDPTPass>(), "UDPT Pass");
			markOutput("UDPT Pass", "Color");
		}

	};
}