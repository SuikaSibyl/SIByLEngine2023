#include <cmath>
#include <memory>
#include <string>
#include <cstdint>
#include <Resource/SE.Core.Resource.hpp>
#include <SE.RHI.hpp>
#include <SE.GFX.hpp>
#include <SE.RDG.hpp>
#include "../Passes/RayTracingPasses/SE.SRenderer-UDPTPass.hpp"

namespace SIByL::SRP
{
SE_EXPORT struct UDPTGraph : public RDG::Graph {
		UDPTGraph() {
			addPass(std::make_unique<UDPTPass>(), "UDPT Pass");
			markOutput("UDPT Pass", "Color");
		}
	};

	SE_EXPORT struct UDPTPipeline : public RDG::SingleGraphPipeline {
		UDPTPipeline() { pGraph = &graph; }
		UDPTGraph graph;
	};
}