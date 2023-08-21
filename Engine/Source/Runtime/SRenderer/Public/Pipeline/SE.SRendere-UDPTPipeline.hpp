#include <cmath>
#include <memory>
#include <string>
#include <cstdint>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include "../Passes/RayTracingPasses/SE.SRenderer-UDPTPass.hpp"
#include "../Passes/FullScreenPasses/SE.SRenderer-AccumulatePass.hpp"

namespace SIByL::SRP
{
SE_EXPORT struct UDPTGraph : public RDG::Graph {
		UDPTGraph() {
			addPass(std::make_unique<UDPTPass>(), "UDPT Pass");
			addPass(std::make_unique<AccumulatePass>(), "Accum Pass");

			addEdge("UDPT Pass", "Color", "Accum Pass", "Input");

			markOutput("Accum Pass", "Output");
		}
	};

	SE_EXPORT struct UDPTPipeline : public RDG::SingleGraphPipeline {
		UDPTPipeline() { pGraph = &graph; }
		UDPTGraph graph;
	};
}