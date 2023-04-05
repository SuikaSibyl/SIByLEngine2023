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
	export struct UDPTGraph :public RDG::Graph {
		UDPTGraph() {
			addPass(std::make_unique<UDPTPass>(), "UDPT Pass");
			markOutput("UDPT Pass", "Color");
		}
	};

	export struct UDPTPipeline :public RDG::SingleGraphPipeline {
		UDPTPipeline() { pGraph = &graph; }
		UDPTGraph graph;
	};
}