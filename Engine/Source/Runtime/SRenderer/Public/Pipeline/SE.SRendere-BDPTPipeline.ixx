module;
#include <cmath>
#include <memory>
#include <string>
#include <cstdint>
export module SE.SRenderer.BDPTPipeline;
import SE.Core.Resource;
import SE.RHI;
import SE.GFX;
import SE.RDG;

import SE.SRenderer.ClearI32RGBA;
import SE.SRenderer.BDPTPass;
import SE.SRenderer.CombineI32RGBA;

namespace SIByL::SRP
{
	export struct BDPTGraph :public RDG::Graph {

		BDPTGraph() {
			addPass(std::make_unique<ClearI32RGBAPass>(), "Clear Pass");
			addPass(std::make_unique<BDPTPass>(), "BDPT Pass");
			addPass(std::make_unique<CombineI32RGBAPass>(), "Combine Pass");

			addEdge("Clear Pass", "I32RGBA", "BDPT Pass", "atomicRGBA");
			addEdge("BDPT Pass", "atomicRGBA", "Combine Pass", "I32RGBA");

			markOutput("Combine Pass", "HDRAccum");
		}
	};

	export struct BDPTPipeline :public RDG::SingleGraphPipeline {
		BDPTPipeline() { pGraph = &graph; }
		BDPTGraph graph;
	};
}