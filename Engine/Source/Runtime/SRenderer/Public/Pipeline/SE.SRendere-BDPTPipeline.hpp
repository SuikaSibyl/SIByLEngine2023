#include <cmath>
#include <memory>
#include <string>
#include <cstdint>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include "../Passes/RayTracingPasses/SE.SRenderer-BDPTPass.hpp"
#include "../Passes/FullScreenPasses/SE.SRenderer-ClearI32RGBA.hpp"
#include "../Passes/FullScreenPasses/SE.SRenderer-CombineI32RGBA.hpp"

namespace SIByL::SRP
{
	SE_EXPORT struct BDPTGraph :public RDG::Graph {

		BDPTGraph() {
			addPass(std::make_unique<ClearI32RGBAPass>(), "Clear Pass");
			addPass(std::make_unique<BDPTPass>(), "BDPT Pass");
			addPass(std::make_unique<CombineI32RGBAPass>(), "Combine Pass");

			addEdge("Clear Pass", "I32RGBA", "BDPT Pass", "atomicRGBA");
			addEdge("BDPT Pass", "atomicRGBA", "Combine Pass", "I32RGBA");

			markOutput("Combine Pass", "HDRAccum");
		}
	};

	SE_EXPORT struct BDPTPipeline :public RDG::SingleGraphPipeline {
		BDPTPipeline() { pGraph = &graph; }
		BDPTGraph graph;
	};
}